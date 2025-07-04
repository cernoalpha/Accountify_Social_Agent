from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback

from ai_sales_agent import (
    AISalesAgent, 
    ClientProfile, 
    CompanyData,
    send_initial_message,
    handle_reply,
    get_conversation_status
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the agent (you might want to make this a singleton or use dependency injection)
agent = AISalesAgent()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Sales Agent API"
    })

@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """
    Start a new conversation with a client
    
    Expected JSON payload:
    {
        "client": {
            "client_id": "client_001",
            "name": "John Doe",
            "email": "john@company.com",
            "company": "TechCorp",
            "position": "CTO",
            "industry": "technology",
            "linkedin_url": "https://linkedin.com/in/johndoe",  # optional
            "twitter_handle": "johndoe"  # optional
        },
        "social_posts": [  # optional
            {
                "platform": "linkedin",
                "content": "Post content...",
                "timestamp": "2024-01-15T10:30:00Z",
                "engagement_metrics": {"likes": 45, "comments": 12},
                "url": "https://linkedin.com/post/123"  # optional
            }
        ],
        "company_data": {  # optional
            "name": "TechCorp",
            "industry": "technology",
            "size": "200-500 employees",
            "revenue": "$50M-100M",  # optional
            "description": "Company description...",
            "recent_news": ["News item 1", "News item 2"],  # optional
            "technologies": ["Python", "React", "AWS"]  # optional
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        # Validate required fields
        if 'client' not in data:
            return jsonify({
                "error": "Client information is required"
            }), 400
        
        client_data = data['client']
        required_client_fields = ['client_id', 'name', 'email', 'company', 'position', 'industry']
        
        for field in required_client_fields:
            if field not in client_data:
                return jsonify({
                    "error": f"Missing required client field: {field}"
                }), 400
        
        # Create client profile
        client_profile = ClientProfile(
            client_id=client_data['client_id'],
            name=client_data['name'],
            email=client_data['email'],
            company=client_data['company'],
            position=client_data['position'],
            industry=client_data['industry'],
            linkedin_url=client_data.get('linkedin_url'),
            twitter_handle=client_data.get('twitter_handle')
        )
        
        # Get social posts data (optional)
        social_posts_data = data.get('social_posts', [])
        
        # Create company data (optional)
        company_data = None
        if 'company_data' in data:
            company_info = data['company_data']
            company_data = CompanyData(
                name=company_info.get('name', client_data['company']),
                industry=company_info.get('industry', client_data['industry']),
                size=company_info.get('size', 'Unknown'),
                revenue=company_info.get('revenue'),
                description=company_info.get('description', f"{client_data['company']} is a company in {client_data['industry']}"),
                recent_news=company_info.get('recent_news', []),
                technologies=company_info.get('technologies', [])
            )
        
        # Generate initial message
        result = asyncio.run(agent.generate_initial_message(
            client_profile=client_profile,
            social_posts_data=social_posts_data,
            company_data=company_data
        ))
        
        return jsonify({
            "success": True,
            "data": {
                "message": result['message'],
                "conversation_id": result['conversation_id'],
                "client_id": result['client_id']
            }
        })
    
    except Exception as e:
        print(f"Error in start_conversation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/conversation/<conversation_id>/reply', methods=['POST'])
def handle_conversation_reply(conversation_id: str):
    """
    Handle a reply in an existing conversation
    
    Expected JSON payload:
    {
        "message": "Client's reply message"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Message is required"
            }), 400
        
        incoming_message = data['message']
        
        # Generate reply
        result = agent.generate_reply(conversation_id, incoming_message)
        
        if "error" in result:
            return jsonify({
                "error": result['error']
            }), 404
        
        return jsonify({
            "success": True,
            "data": {
                "reply": result['reply'],
                "conversation_id": conversation_id,
                "conversation_stage": result['conversation_stage'],
                "client_interests": result['client_interests'],
                "products_mentioned": result['products_mentioned'],
                "message_count": result['message_count']
            }
        })
    
    except Exception as e:
        print(f"Error in handle_conversation_reply: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/conversation/<conversation_id>/summary', methods=['GET'])
def get_conversation_summary(conversation_id: str):
    """
    Get summary of a conversation
    """
    try:
        summary = agent.get_conversation_summary(conversation_id)
        
        if not summary:
            return jsonify({
                "error": "Conversation not found"
            }), 404
        
        return jsonify({
            "success": True,
            "data": summary
        })
    
    except Exception as e:
        print(f"Error in get_conversation_summary: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation_details(conversation_id: str):
    """
    Get detailed conversation history
    """
    try:
        conversation = agent.context_manager.get_conversation_history(conversation_id)
        
        if not conversation:
            return jsonify({
                "error": "Conversation not found"
            }), 404
        
        # Get client context for additional info
        client_context = agent.context_manager.get_client_context(conversation.client_id)
        client_name = "Unknown"
        if client_context and 'profile' in client_context:
            client_name = client_context['profile'].get('name', 'Unknown')
        
        return jsonify({
            "success": True,
            "data": {
                "conversation_id": conversation.conversation_id,
                "client_id": conversation.client_id,
                "client_name": client_name,
                "messages": conversation.messages,
                "conversation_stage": conversation.conversation_stage,
                "client_interests": conversation.client_interests,
                "products_mentioned": conversation.products_mentioned
            }
        })
    
    except Exception as e:
        print(f"Error in get_conversation_details: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """
    List all conversations (with pagination support)
    Query parameters:
    - limit: Number of conversations to return (default: 50)
    - offset: Number of conversations to skip (default: 0)
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # This would require adding a method to context_manager to list conversations
        # For now, return a placeholder response
        return jsonify({
            "success": True,
            "data": {
                "conversations": [],  # Would be populated with actual data
                "total": 0,
                "limit": limit,
                "offset": offset
            },
            "message": "List conversations endpoint - implementation pending"
        })
    
    except Exception as e:
        print(f"Error in list_conversations: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    print("Starting AI Sales Agent API...")
    print("Available endpoints:")
    print("- POST /api/conversation/start - Start a new conversation")
    print("- POST /api/conversation/<id>/reply - Reply to a conversation")
    print("- GET /api/conversation/<id>/summary - Get conversation summary")
    print("- GET /api/conversation/<id> - Get full conversation details")
    print("- GET /api/conversations - List all conversations")
    print("- GET /health - Health check")
    
    app.run(debug=True)