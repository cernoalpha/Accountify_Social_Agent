# AI Sales Agent API Documentation

RESTful API service for managing AI-powered sales conversations. The API supports starting conversations, handling replies, and retrieving conversation details.

## Base URL
http://localhost:5000


## Endpoints

### Health Check
`GET /health`

Check API service health status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-03-28T10:30:00.000Z",
    "service": "AI Sales Agent API"
}
```

### Start Conversation

POST /api/conversation/start

Initialize a new sales conversation.
Request Body:
```json
{
    "client": {
        "client_id": "string",
        "name": "string",
        "email": "string",
        "company": "string",
        "position": "string",
        "industry": "string",
        "linkedin_url": "string (optional)",
        "twitter_handle": "string (optional)"
    },
    "social_posts": [
        {
            "platform": "string",
            "content": "string",
            "timestamp": "ISO datetime string",
            "engagement_metrics": {
                "likes": "number",
                "comments": "number"
            },
            "url": "string (optional)"
        }
    ],
    "company_data": {
        "name": "string",
        "industry": "string",
        "size": "string",
        "revenue": "string (optional)",
        "description": "string",
        "recent_news": ["string"],
        "technologies": ["string"]
    },
    "products_services": {
        "product_key": {
            "name": "string",
            "description": "string",
            "target_industries": ["string"],
            "pain_points": ["string"]
        }
    },
    "system_prompt": "string"
}
```
Response:
```json
{
    "success": true,
    "data": {
        "message": "string",
        "conversation_id": "string",
        "client_id": "string"
    }
}
```

### Handle Reply
POST /api/conversation/{conversation_id}/reply

Handle a reply in an existing conversation.

Request Body:
```json
{
    "message": "string",
    "products_services": {
        "product_key": {
            "name": "string",
            "description": "string",
            "target_industries": ["string"],
            "pain_points": ["string"]
        }
    },
    "reply_system_prompt": "string"
}
```

Response:
```json
{
    "success": true,
    "data": {
        "reply": "string",
        "conversation_id": "string",
        "conversation_stage": "string",
        "client_interests": ["string"],
        "products_mentioned": ["string"],
        "message_count": "number"
    }
}
```

### Get Conversation Summary
GET /api/conversation/{conversation_id}/summary

Retrieve a summary of the conversation.

Response:
```json
{
    "success": true,
    "data": {
        "conversation_id": "string",
        "client_id": "string",
        "client_name": "string",
        "conversation_stage": "string",
        "message_count": "number",
        "last_message_sent": "string",
        "client_interests": ["string"],
        "products_mentioned": ["string"],
        "last_updated": "ISO datetime string"
    }
}
```

### Get Conversation Details
GET /api/conversation/{conversation_id}

Get detailed conversation history.

Response:
```json
{
    "success": true,
    "data": {
        "conversation_id": "string",
        "client_id": "string",
        "client_name": "string",
        "messages": [
            {
                "role": "string",
                "content": "string",
                "timestamp": "ISO datetime string"
            }
        ],
        "conversation_stage": "string",
        "client_interests": ["string"],
        "products_mentioned": ["string"]
    }
}
```