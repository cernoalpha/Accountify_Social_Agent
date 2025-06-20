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
            existing_profile = db.query(ClientProfiles).filter(
                ClientProfiles.client_id == client_profile.client_id
            ).first()
            profile_data = json.dumps({
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
            if existing_profile:
                # Update existing profile
                existing_profile.profile_data = profile_data
                existing_profile.last_updated = datetime.utcnow()
            else:
                # Insert new profile
                profile_record = ClientProfiles(
                    client_id=client_profile.client_id,
                    profile_data=profile_data,
                    last_updated=datetime.utcnow()
                )
                db.add(profile_record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error storing client profile: {e}")
            raise
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
            # Check if conversation exists
            existing_conv = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_context.conversation_id
            ).first()
            
            if existing_conv:
                # Update existing conversation
                existing_conv.messages = json.dumps(conversation_context.messages)
                existing_conv.stage = conversation_context.conversation_stage
                existing_conv.last_updated = datetime.utcnow()
            else:
                # Create new conversation
                conv_record = ConversationHistory(
                    conversation_id=conversation_context.conversation_id,
                    client_id=conversation_context.client_id,
                    messages=json.dumps(conversation_context.messages),
                    stage=conversation_context.conversation_stage
                )
                db.add(conv_record)
            
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error storing conversation: {e}")
            raise
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
            for post in social_posts[:3]:  # Include last 3 posts
                recent_posts_text += f"- [{post.platform}] {post.content[:150]}...\n"
        
        company_news = ""
        if company_data.recent_news:
            company_news = "\nRecent company news:\n- " + "\n- ".join(company_data.recent_news[:2])
        
        return f"""
        Please analyze this prospect's profile and generate a personalized outreach message:
        
        Name: {client_profile.name}
        Position: {client_profile.position}
        Company: {client_profile.company}
        Industry: {client_profile.industry}
        
        Company Details:
        Size: {company_data.size}
        Description: {company_data.description}
        Technologies: {', '.join(company_data.technologies)}
        {company_news}
        {recent_posts_text}
        
        Consider their industry challenges and our relevant solutions:
        {json.dumps(self.products_services, indent=2)}
        """

    def _build_reply_prompt(self, conversation: ConversationContext, 
                          incoming_message: str, 
                          client_context: Dict) -> str:
        """Build prompt for generating contextual replies"""
        
        # Get last 5 messages for context
        recent_messages = conversation.messages[-5:] if len(conversation.messages) > 5 else conversation.messages
        conversation_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in recent_messages
        ])
        
        client_info = "Limited context available"
        if client_context:
            profile = client_context.get('profile', {})
            company = client_context.get('company_data', {})
            client_info = f"""
            Name: {profile.get('name')}
            Position: {profile.get('position')} at {profile.get('company')}
            Industry: {profile.get('industry')}
            Company Info: {company.get('description', 'Not available')}
            """

        return f"""
        Previous conversation:
        {conversation_history}
        
        Client Information:
        {client_info}
        
        Current conversation stage: {conversation.conversation_stage}
        Client's interests identified: {', '.join(conversation.client_interests)}
        Products discussed: {', '.join(conversation.products_mentioned)}
        
        New message from client: "{incoming_message}"
        
        Available solutions: {json.dumps(self.products_services, indent=2)}
        """

    def _analyze_conversation_stage(self, message: str, 
                                  conversation_history: List[Dict]) -> str:
        """Determine conversation stage based on message content and history"""
        
        message_lower = message.lower()
        
        # Define stage transition keywords
        stage_indicators = {
            'engaged': ['interested', 'tell me more', 'sounds good', 'learn more',
                       'when can we', 'schedule', 'meet'],
            'negotiating': ['price', 'cost', 'budget', 'proposal', 'quote',
                          'investment', 'terms'],
            'closing': ['yes', 'agreed', "let's proceed", 'move forward',
                       'contract', 'start'],
            'closed_lost': ['not interested', 'no thanks', 'remove me',
                          'unsubscribe', 'stop']
        }
        
        # Check for stage transitions
        for stage, keywords in stage_indicators.items():
            if any(keyword in message_lower for keyword in keywords):
                return stage
        
        # If no clear indicators, stay in current stage
        return "prospecting"

# Standalone helper functions
def format_timestamp(dt: datetime) -> str:
    """Format datetime for consistent API responses"""
    return dt.isoformat() if isinstance(dt, datetime) else dt

def validate_client_data(client_data: Dict) -> bool:
    """Validate required client data fields"""
    required_fields = ['client_id', 'name', 'email', 'company', 'position', 'industry']
    return all(field in client_data for field in required_fields)

# If running as standalone script
if __name__ == "__main__":
    print("AI Sales Agent module loaded successfully")