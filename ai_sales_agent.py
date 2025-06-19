import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import openai
import chromadb
from chromadb.config import Settings
import requests
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///sales_agent.db")
    CHROMA_PERSIST_DIR = "./chroma_db"

# Data Models
@dataclass
class ClientProfile:
    client_id: str
    name: str
    email: str
    company: str
    position: str
    industry: str
    linkedin_url: Optional[str] = None
    twitter_handle: Optional[str] = None

@dataclass
class SocialPost:
    platform: str
    content: str
    timestamp: datetime
    engagement_metrics: Dict[str, int]
    url: Optional[str] = None

@dataclass
class CompanyData:
    name: str
    industry: str
    size: str
    revenue: Optional[str]
    description: str
    recent_news: List[str]
    technologies: List[str]

@dataclass
class ConversationContext:
    conversation_id: str
    client_id: str
    messages: List[Dict[str, str]]
    last_message_sent: str
    products_mentioned: List[str]
    client_interests: List[str]
    conversation_stage: str  # prospecting, engaged, negotiating, closed

# Database Setup
Base = declarative_base()

class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, unique=True, index=True)
    client_id = Column(String, index=True)
    messages = Column(Text)  # JSON string
    last_updated = Column(DateTime, default=datetime.utcnow)
    stage = Column(String, default="prospecting")

class ClientProfiles(Base):
    __tablename__ = "client_profiles"
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String, unique=True, index=True)
    profile_data = Column(Text)  # JSON string
    last_updated = Column(DateTime, default=datetime.utcnow)

# Helper function to create SocialPost from external data
def create_social_posts_from_data(posts_data: List[Dict[str, Any]]) -> List[SocialPost]:
    """
    Convert external social media data to SocialPost objects
    Expected format:
    [
        {
            "platform": "linkedin",
            "content": "Excited to announce...",
            "timestamp": "2024-01-15T10:30:00Z",
            "engagement_metrics": {"likes": 45, "comments": 12, "shares": 8},
            "url": "https://linkedin.com/post/123"
        }
    ]
    """
    posts = []
    for post_data in posts_data:
        timestamp = post_data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.utcnow()
        
        post = SocialPost(
            platform=post_data.get('platform', 'unknown'),
            content=post_data.get('content', ''),
            timestamp=timestamp,
            engagement_metrics=post_data.get('engagement_metrics', {}),
            url=post_data.get('url')
        )
        posts.append(post)
    return posts

# Context Management System
class ContextManager:
    def __init__(self):
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="client_context",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize SQL database for structured data
        self.engine = create_engine(Config.DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def store_client_context(self, client_profile: ClientProfile, 
                           social_posts: List[SocialPost], 
                           company_data: CompanyData):
        """Store client context in both vector and relational databases"""
        
        # Prepare text for vector embedding
        context_text = self._prepare_context_text(client_profile, social_posts, company_data)
        
        # Store in vector database for semantic search
        self.collection.add(
            documents=[context_text],
            metadatas=[{
                "client_id": client_profile.client_id,
                "type": "full_context",
                "timestamp": datetime.utcnow().isoformat()
            }],
            ids=[f"context_{client_profile.client_id}"]
        )
        
        # Store structured data in SQL database
        db = self.SessionLocal()
        try:
            profile_record = ClientProfiles(
                client_id=client_profile.client_id,
               profile_data=json.dumps({
        "profile": asdict(client_profile),
        "social_posts": [
            {
                **asdict(post),
                "timestamp": post.timestamp.isoformat() if isinstance(post.timestamp, datetime) else post.timestamp
            }
            for post in social_posts
        ],
        "company_data": asdict(company_data)
    })
)
            db.merge(profile_record)
            db.commit()
        finally:
            db.close()
    
    def _prepare_context_text(self, client_profile: ClientProfile, 
                            social_posts: List[SocialPost], 
                            company_data: CompanyData) -> str:
        """Prepare comprehensive context text for vector embedding"""
        
        context_parts = [
            f"Client: {client_profile.name}",
            f"Position: {client_profile.position} at {client_profile.company}",
            f"Industry: {client_profile.industry}",
            f"Company: {company_data.name} - {company_data.description}",
            f"Company Size: {company_data.size}",
            f"Technologies: {', '.join(company_data.technologies)}",
        ]
        
        # Add recent social posts
        context_parts.append("Recent Social Activity:")
        for post in social_posts[-5:]:  # Last 5 posts
            context_parts.append(f"[{post.platform}] {post.content[:200]}...")
        
        # Add company news
        if company_data.recent_news:
            context_parts.append("Recent Company News:")
            context_parts.extend(company_data.recent_news[:3])
        
        return "\n".join(context_parts)
    
    def get_client_context(self, client_id: str) -> Optional[Dict]:
        """Retrieve client context from database"""
        db = self.SessionLocal()
        try:
            record = db.query(ClientProfiles).filter(
                ClientProfiles.client_id == client_id
            ).first()
            if record:
                return json.loads(record.profile_data)
            return None
        finally:
            db.close()
    
    def store_conversation(self, conversation_context: ConversationContext):
        """Store conversation history"""
        db = self.SessionLocal()
        try:
            conv_record = ConversationHistory(
                conversation_id=conversation_context.conversation_id,
                client_id=conversation_context.client_id,
                messages=json.dumps(conversation_context.messages),
                stage=conversation_context.conversation_stage
            )
            db.merge(conv_record)
            db.commit()
        finally:
            db.close()
    
    def get_conversation_history(self, conversation_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation history"""
        db = self.SessionLocal()
        try:
            record = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id
            ).first()
            if record:
                messages = json.loads(record.messages)
                return ConversationContext(
                    conversation_id=record.conversation_id,
                    client_id=record.client_id,
                    messages=messages,
                    last_message_sent=messages[-1]["content"] if messages else "",
                    products_mentioned=[],  # Would extract from messages
                    client_interests=[],    # Would extract from messages
                    conversation_stage=record.stage
                )
            return None
        finally:
            db.close()

# AI Sales Agent
class AISalesAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.context_manager = ContextManager()
        
        # Your product/service catalog
        self.products_services = {
            "ai_consulting": {
                "name": "AI Consulting Services",
                "description": "Strategic AI implementation and transformation consulting",
                "target_industries": ["technology", "finance", "healthcare", "retail"],
                "pain_points": ["digital transformation", "automation", "efficiency", "innovation"]
            },
            "data_analytics": {
                "name": "Advanced Data Analytics Platform",
                "description": "Enterprise-grade data analytics and business intelligence",
                "target_industries": ["finance", "retail", "manufacturing", "healthcare"],
                "pain_points": ["data insights", "decision making", "reporting", "analytics"]
            }
        }
    
    async def generate_initial_message(self, client_profile: ClientProfile, 
                                     social_posts_data: List[Dict[str, Any]] = None,
                                     company_data: CompanyData = None) -> Dict[str, str]:
        """
        Generate personalized initial outreach message
        
        Args:
            client_profile: Basic client information
            social_posts_data: List of social media posts data
            company_data: Company information (optional)
            
        Returns:
            Dict containing message and conversation_id for future reference
        """
        
        # Convert social media data to SocialPost objects
        social_posts = []
        if social_posts_data:
            social_posts = create_social_posts_from_data(social_posts_data)
        
        # Use provided company data or create basic one
        if not company_data:
            company_data = CompanyData(
                name=client_profile.company,
                industry=client_profile.industry,
                size="Unknown",
                revenue=None,
                description=f"{client_profile.company} is a company in {client_profile.industry}",
                recent_news=[],
                technologies=[]
            )
        
        # Store context
        self.context_manager.store_client_context(client_profile, social_posts, company_data)
        
        # Generate personalized message
        context_prompt = self._build_context_prompt(client_profile, social_posts, company_data)
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert sales professional who creates highly personalized, 
                    non-salesy outreach messages. Your goal is to start meaningful conversations by 
                    referencing specific, relevant details about the prospect's recent activity, 
                    company situation, or industry challenges. Keep messages concise (2-3 sentences), 
                    professional but warm, and always include a soft call-to-action."""
                },
                {
                    "role": "user",
                    "content": context_prompt
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        generated_message = response.choices[0].message.content
        
        # Create unique conversation ID
        conversation_id = f"conv_{client_profile.client_id}_{int(datetime.utcnow().timestamp())}"
        
        # Store initial conversation
        conversation_context = ConversationContext(
            conversation_id=conversation_id,
            client_id=client_profile.client_id,
            messages=[{
                "role": "assistant", 
                "content": generated_message,
                "timestamp": datetime.utcnow().isoformat()
            }],
            last_message_sent=generated_message,
            products_mentioned=[],
            client_interests=[],
            conversation_stage="prospecting"
        )
        
        self.context_manager.store_conversation(conversation_context)
        
        return {
            "message": generated_message,
            "conversation_id": conversation_id,
            "client_id": client_profile.client_id
        }
    
    def generate_reply(self, conversation_id: str, incoming_message: str) -> Dict[str, Any]:
        """
        Generate contextual reply based on conversation history
        This method can be called anytime - minutes, hours, or days later
        
        Args:
            conversation_id: Unique conversation identifier from initial message
            incoming_message: The client's reply message
            
        Returns:
            Dict containing the reply and updated conversation info
        """
        
        # Get conversation history
        conversation = self.context_manager.get_conversation_history(conversation_id)
        if not conversation:
            return {
                "error": "Conversation not found",
                "message": "I apologize, but I don't have the context of our previous conversation. Could you please provide more details?"
            }
        
        # Get client context
        client_context = self.context_manager.get_client_context(conversation.client_id)
        
        # Build conversation prompt
        conversation_prompt = self._build_reply_prompt(
            conversation, incoming_message, client_context
        )
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional sales representative continuing a conversation. 
                    Respond naturally and helpfully to the prospect's message. Reference previous 
                    conversation context appropriately. If they show interest, provide relevant 
                    information about your solutions. If they have questions, answer them thoroughly. 
                    Always maintain a consultative, helpful tone rather than being pushy."""
                },
                {
                    "role": "user",
                    "content": conversation_prompt
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        reply = response.choices[0].message.content
        
        # Update conversation history with timestamps
        current_time = datetime.utcnow().isoformat()
        conversation.messages.extend([
            {
                "role": "user", 
                "content": incoming_message,
                "timestamp": current_time
            },
            {
                "role": "assistant", 
                "content": reply,
                "timestamp": current_time
            }
        ])
        conversation.last_message_sent = reply
        
        # Update conversation stage based on reply sentiment
        conversation.conversation_stage = self._analyze_conversation_stage(
            incoming_message, conversation.messages
        )
        
        # Extract interests and mentioned products for better context
        conversation.client_interests = self._extract_interests(incoming_message, conversation.messages)
        conversation.products_mentioned = self._extract_mentioned_products(conversation.messages)
        
        self.context_manager.store_conversation(conversation)
        
        return {
            "reply": reply,
            "conversation_id": conversation_id,
            "conversation_stage": conversation.conversation_stage,
            "client_interests": conversation.client_interests,
            "products_mentioned": conversation.products_mentioned,
            "message_count": len(conversation.messages)
        }
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the conversation for reference
        Useful for understanding conversation status when replies come later
        """
        conversation = self.context_manager.get_conversation_history(conversation_id)
        if not conversation:
            return None
        
        client_context = self.context_manager.get_client_context(conversation.client_id)
        client_name = "Unknown"
        if client_context and 'profile' in client_context:
            client_name = client_context['profile'].get('name', 'Unknown')
        
        return {
            "conversation_id": conversation_id,
            "client_id": conversation.client_id,
            "client_name": client_name,
            "conversation_stage": conversation.conversation_stage,
            "message_count": len(conversation.messages),
            "last_message_sent": conversation.last_message_sent,
            "client_interests": conversation.client_interests,
            "products_mentioned": conversation.products_mentioned,
            "last_updated": conversation.messages[-1].get('timestamp') if conversation.messages else None
        }
    
    def _extract_interests(self, message: str, conversation_history: List[Dict]) -> List[str]:
        """Extract client interests from conversation"""
        interests = []
        message_lower = message.lower()
        
        # Simple keyword extraction (can be enhanced with NLP)
        interest_keywords = {
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'automation'],
            'data': ['data', 'analytics', 'insights', 'reporting'],
            'efficiency': ['efficiency', 'productivity', 'streamline', 'optimize'],
            'growth': ['growth', 'scale', 'expand', 'revenue'],
            'cost_reduction': ['cost', 'save', 'reduce', 'budget']
        }
        
        for category, keywords in interest_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                interests.append(category)
        
        return list(set(interests))  # Remove duplicates
    
    def _extract_mentioned_products(self, conversation_history: List[Dict]) -> List[str]:
        """Extract products/services mentioned in conversation"""
        mentioned_products = []
        
        for message in conversation_history:
            if message['role'] == 'assistant':
                content_lower = message['content'].lower()
                for product_key, product_info in self.products_services.items():
                    if product_info['name'].lower() in content_lower:
                        mentioned_products.append(product_key)
        
        return list(set(mentioned_products))  # Remove duplicates
    
    def _build_context_prompt(self, client_profile: ClientProfile, 
                            social_posts: List[SocialPost], 
                            company_data: CompanyData) -> str:
        """Build prompt for initial message generation"""
        
        recent_posts_text = ""
        if social_posts:
            recent_posts_text = "\nRecent social media activity:\n"
            for post in social_posts[:3]:
                recent_posts_text += f"- [{post.platform}] {post.content[:150]}...\n"
        
        return f"""
        Generate a personalized outreach message for this prospect:
        
        Name: {client_profile.name}
        Position: {client_profile.position}
        Company: {client_profile.company}
        Industry: {client_profile.industry}
        
        Company Info: {company_data.description}
        {recent_posts_text}
        
        Available solutions that might be relevant:
        {json.dumps(self.products_services, indent=2)}
        
        Create a personalized message that:
        1. References something specific about their recent activity or company
        2. Connects to a relevant business challenge
        3. Suggests a brief, low-pressure conversation
        4. Feels natural and human, not templated
        """
    
    def _build_reply_prompt(self, conversation: ConversationContext, 
                          incoming_message: str, client_context: Dict) -> str:
        """Build prompt for reply generation"""
        
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation.messages[-6:]
        ])
        
        return f"""
        Previous conversation:
        {conversation_history}
        
        Client context: {json.dumps(client_context, indent=2) if client_context else 'Limited context available'}
        
        New message from prospect: "{incoming_message}"
        
        Current conversation stage: {conversation.conversation_stage}
        
        Available solutions: {json.dumps(self.products_services, indent=2)}
        
        Generate an appropriate response that:
        1. Acknowledges their message thoughtfully
        2. Provides helpful information if they asked questions
        3. Moves the conversation forward naturally
        4. References relevant solutions when appropriate
        5. Maintains the relationship-building tone
        """
    
    def _analyze_conversation_stage(self, message: str, conversation_history: List[Dict]) -> str:
        """Analyze conversation stage based on message content"""
        
        message_lower = message.lower()
        
        # Simple keyword-based stage detection (could be enhanced with ML)
        if any(word in message_lower for word in ['interested', 'tell me more', 'sounds good', 'when can we']):
            return 'engaged'
        elif any(word in message_lower for word in ['price', 'cost', 'budget', 'proposal']):
            return 'negotiating'
        elif any(word in message_lower for word in ['yes', 'agreed', 'let\'s proceed', 'move forward']):
            return 'closing'
        elif any(word in message_lower for word in ['not interested', 'no thanks', 'remove me']):
            return 'closed_lost'
        else:
            return 'prospecting'

# Usage Example - Updated for external data input and delayed conversations
async def main():
    # Initialize the AI Sales Agent
    agent = AISalesAgent()
    
    # Example client profile
    client = ClientProfile(
        client_id="client_001",
        name="Sarah Johnson",
        email="sarah.johnson@innovatetech.com",
        company="InnovateTech Solutions",
        position="Chief Technology Officer",
        industry="technology",
        linkedin_url="https://linkedin.com/in/sarahjohnson",
        twitter_handle="sarah_j_tech"
    )
    
    # Example social media posts data (you provide this from your scraper)
    social_posts_data = [
        {
            "platform": "linkedin",
            "content": "Excited to announce our company's digital transformation initiative! We're looking to implement AI solutions across our operations to improve efficiency and customer experience. Any recommendations for reliable AI consulting partners? #DigitalTransformation #AI",
            "timestamp": "2024-01-15T10:30:00Z",
            "engagement_metrics": {"likes": 47, "comments": 15, "shares": 8},
            "url": "https://linkedin.com/post/123456"
        },
        {
            "platform": "twitter",
            "content": "Frustrated with our current data analytics setup. We have tons of data but struggle to get actionable insights. Time for an upgrade! 📊 #DataAnalytics #BusinessIntelligence",
            "timestamp": "2024-01-12T14:22:00Z",
            "engagement_metrics": {"likes": 23, "retweets": 5, "replies": 12},
            "url": "https://twitter.com/sarah_j_tech/status/789012"
        }
    ]
    
    # Example company data
    company_data = CompanyData(
        name="InnovateTech Solutions",
        industry="technology",
        size="Mid-size (200-500 employees)",
        revenue="$50M-100M",
        description="InnovateTech Solutions provides enterprise software solutions for mid-market companies, focusing on workflow automation and business process optimization.",
        recent_news=[
            "InnovateTech closes Series B funding round of $25M",
            "Company expands to European markets",
            "New partnership with Microsoft Azure announced"
        ],
        technologies=["Microsoft .NET", "Azure", "SQL Server", "React", "Python"]
    )
    
    print("=== INITIAL OUTREACH ===")
    # Generate initial personalized message
    initial_result = await agent.generate_initial_message(
        client_profile=client,
        social_posts_data=social_posts_data,
        company_data=company_data
    )
    
    print(f"Generated Message: {initial_result['message']}")
    print(f"Conversation ID: {initial_result['conversation_id']}")
    
    # Store the conversation ID for later use
    conversation_id = initial_result['conversation_id']
    
    print("\n=== SIMULATING DELAYED REPLY (could be hours/days later) ===")
    
    # Simulate time passing... (in real scenario, this could be hours or days later)
    # You would store the conversation_id and use it when a reply comes in
    
    # Mock incoming reply (this could come from email, LinkedIn, etc.)
    incoming_reply = """Hi there! Thanks for reaching out at the perfect time. I saw your message right after posting about our digital transformation challenges. 
    
    We're definitely looking for AI consulting services, especially around data analytics and process automation. Your timing couldn't be better!
    
    Could you tell me more about your approach to AI implementation and maybe share some case studies similar to our industry?
    
    Best regards,
    Sarah"""
    
    print(f"Client Reply (received later): {incoming_reply}")
    
    # Generate contextual response using the stored conversation_id
    reply_result = agent.generate_reply(conversation_id, incoming_reply)
    
    if "error" not in reply_result:
        print(f"\nAgent Response: {reply_result['reply']}")
        print(f"Conversation Stage: {reply_result['conversation_stage']}")
        print(f"Detected Interests: {reply_result['client_interests']}")
        print(f"Total Messages in Conversation: {reply_result['message_count']}")
    else:
        print(f"Error: {reply_result['error']}")
    
    print("\n=== CONVERSATION SUMMARY ===")
    # Get conversation summary (useful for tracking multiple conversations)
    summary = agent.get_conversation_summary(conversation_id)
    if summary:
        print(f"Client: {summary['client_name']}")
        print(f"Stage: {summary['conversation_stage']}")
        print(f"Messages Exchanged: {summary['message_count']}")
        print(f"Client Interests: {summary['client_interests']}")
        print(f"Products Mentioned: {summary['products_mentioned']}")
    
    print("\n=== ANOTHER DELAYED REPLY ===")
    
    # Simulate another reply coming later
    second_reply = "That sounds great! I'm particularly interested in the data analytics platform you mentioned. Could we schedule a call next week to discuss pricing and implementation timeline?"
    
    print(f"Second Client Reply: {second_reply}")
    
    second_response = agent.generate_reply(conversation_id, second_reply)
    if "error" not in second_response:
        print(f"\nAgent Response: {second_response['reply']}")
        print(f"Updated Stage: {second_response['conversation_stage']}")

# Standalone functions for easy integration
def send_initial_message(client_data: Dict, social_posts: List[Dict], company_info: Dict = None):
    """
    Standalone function to send initial message
    Use this in your main application
    """
    agent = AISalesAgent()
    
    client_profile = ClientProfile(**client_data)
    company_data = None
    if company_info:
        company_data = CompanyData(**company_info)
    
    result = asyncio.run(agent.generate_initial_message(
        client_profile=client_profile,
        social_posts_data=social_posts,
        company_data=company_data
    ))
    
    return result

def handle_reply(conversation_id: str, reply_message: str):
    """
    Standalone function to handle replies
    Use this when you receive a reply (could be anytime later)
    """
    agent = AISalesAgent()
    return agent.generate_reply(conversation_id, reply_message)

def get_conversation_status(conversation_id: str):
    """
    Get current conversation status
    """
    agent = AISalesAgent()
    return agent.get_conversation_summary(conversation_id)

if __name__ == "__main__":
    asyncio.run(main())
