#!/usr/bin/env python3
"""
Simple FastAPI demo for Customer Support Agent
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import uuid
from dotenv import load_dotenv

# Import from our graph_build module
from graph_build import (
    classify_inquiry, 
    interactive_graph, 
    customers,
    specialists,
    run_web_support_session
)
from langgraph.types import Command

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Customer Support Agent Demo",
    description="Simple demo API for the LangGraph Customer Support Agent",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware for frontend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ClassifyRequest(BaseModel):
    customer_question: str

class ClassifyResponse(BaseModel):
    category: str
    specialist: Dict[str, Any]

class SupportRequest(BaseModel):
    customer_id: str
    customer_question: str

class SupportResponse(BaseModel):
    customer_id: str
    customer_question: str
    response: str
    escalated: bool
    specialist_info: Optional[Dict[str, Any]] = None
    search_scores: Dict[str, float]

class ChatRequest(BaseModel):
    customer_id: str
    message: str

class ChatResponse(BaseModel):
    success: bool
    customer_id: str
    message: str
    response: str
    is_escalated: bool
    specialist_info: Optional[Dict[str, Any]] = None
    search_quality: Dict[str, Any]
    customer_context: str
    error: Optional[str] = None

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# API root
@app.get("/api")
async def root():
    return {"message": "Customer Support Agent API", "status": "running"}

@app.get("/health")
async def health():
    print("🔍 Health check endpoint called")
    return {"status": "ok"}

# Classification endpoint
@app.post("/classify", response_model=ClassifyResponse)
async def classify_question(request: ClassifyRequest):
    """Classify a customer question into support categories."""
    print(f"📥 Classification request: {request.customer_question}")
    try:
        category = classify_inquiry(request.customer_question)
        print(f"🏷️ Classified as: {category}")
        specialist_info = specialists.get(category, specialists["General"])
        
        response = ClassifyResponse(
            category=category,
            specialist=specialist_info
        )
        print(f"📤 Sending response: {response}")
        return response
    except Exception as e:
        print(f"❌ Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Main support endpoint (simplified version without interrupts)
@app.post("/support", response_model=SupportResponse)
async def handle_support_request(request: SupportRequest):
    """Handle a complete customer support request."""
    try:
        # Validate customer ID
        if request.customer_id not in customers:
            raise HTTPException(status_code=404, detail="Customer ID not found")
        
        # Create a unique thread for this request
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
        
        # Simulate the full workflow but without interactive interrupts
        # We'll use the graph in a non-interactive way
        initial_state = {
            "customer_question": request.customer_question,
            "customer_id": request.customer_id,
            "customer_context": "",
            "knowledge_base_results": [],
            "web_search_raw_results": [],
            "specialist_info": {},
            "messages": []
        }
        
        # For demo purposes, we'll simulate the workflow steps
        # In a real implementation, you'd want to modify the graph to be non-interactive
        
        # Get customer context
        customer_data = customers[request.customer_id]
        customer_context = f"""Customer Information:
Name: {customer_data['name']}
Email: {customer_data['email']}
Plan: {customer_data['plan']}
Status: {customer_data['subscription_status']}
Account Age: {customer_data['account_age_days']} days
Previous Issues: {customer_data['previous_issues']}"""
        
        # Classify the inquiry
        category = classify_inquiry(request.customer_question)
        
        # For demo, we'll create a simple response
        # In production, you'd run the actual graph workflow
        escalated = category in ["Urgent"] or "critical" in request.customer_question.lower()
        
        if escalated:
            specialist_info = specialists.get(category, specialists["General"])
            response_text = f"""I'm escalating your inquiry to our specialist team.

🏷️ **Category**: {category}
👤 **Assigned Specialist**: {specialist_info['specialist']}
📧 **Email**: {specialist_info['email']}
⏱️ **Expected Response Time**: {specialist_info['response_time']}

They will contact you shortly with a comprehensive solution."""
        else:
            specialist_info = None
            response_text = f"""Thank you for contacting support, {customer_data['name']}!

🏷️ **Category**: {category}
📋 **Your Plan**: {customer_data['plan']} ({customer_data['subscription_status']})

Based on your inquiry about "{request.customer_question}", I've found relevant information in our knowledge base. For detailed assistance, please refer to our documentation or contact our {category.lower()} support team.

Is there anything else I can help you with today?"""
        
        return SupportResponse(
            customer_id=request.customer_id,
            customer_question=request.customer_question,
            response=response_text,
            escalated=escalated,
            specialist_info=specialist_info,
            search_scores={
                "knowledge_base": 0.75,  # Mock scores for demo
                "web_search": 0.60
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Support request failed: {str(e)}")

# Full chatbot endpoint using the complete graph workflow
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Handle a complete customer support conversation using the full graph workflow."""
    print(f"💬 Chat request from {request.customer_id}: {request.message}")
    
    try:
        # Validate customer ID
        if request.customer_id not in customers:
            raise HTTPException(status_code=404, detail="Customer ID not found")
        
        # Run the full graph workflow
        result = run_web_support_session(request.customer_id, request.message)
        
        if result["success"]:
            return ChatResponse(
                success=True,
                customer_id=result["customer_id"],
                message=result["customer_question"],
                response=result["response"],
                is_escalated=result["is_escalated"],
                specialist_info=result.get("specialist_info"),
                search_quality=result["search_quality"],
                customer_context=result["customer_context"]
            )
        else:
            return ChatResponse(
                success=False,
                customer_id=request.customer_id,
                message=request.message,
                response="I apologize, but I encountered an error while processing your request.",
                is_escalated=False,
                specialist_info=None,
                search_quality={"knowledge_base_score": 0.0, "web_search_score": 0.0},
                customer_context="",
                error=result.get("error", "Unknown error")
            )
            
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")

# List available customers (for demo purposes)
@app.get("/customers")
async def list_customers():
    """List available customer IDs for testing."""
    return {
        "customers": list(customers.keys()),
        "sample_questions": [
            "How do I reset my password?",
            "I'm having API connection issues",
            "I need help with billing",
            "What are your service hours?",
            "Critical system outage!"
        ]
    }

# List specialists
@app.get("/specialists")
async def list_specialists():
    """List available specialists and their categories."""
    return {"specialists": specialists}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
