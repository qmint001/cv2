import os
import asyncio
import dotenv
from flask import Flask, render_template, request, jsonify
from agents import Agent, Runner
from typing import List, Dict, Any

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Create our specialized agents
research_agent = Agent(
    name="Research Agent",
    handoff_description="Specialist agent for gathering travel information about destinations",
    instructions="""You are a travel research specialist. Your goal is to provide comprehensive, accurate information about travel destinations.
    
    Focus on gathering and presenting information about:
    - Key attractions and points of interest
    - Accommodation options
    - Local transportation
    - Weather and best times to visit
    - Cultural norms and customs
    - Travel advisories or safety information
    
    Always cite your sources if possible. Be objective and thorough in your research.
    """
)

planning_agent = Agent(
    name="Planning Agent",
    handoff_description="Specialist agent for creating detailed travel plans and itineraries",
    instructions="""You are a travel planning specialist. Your goal is to create detailed, personalized travel itineraries.
    
    Your plans should include:
    - Day-by-day activity suggestions
    - Accommodation recommendations
    - Transportation logistics
    - Estimated budget
    - Packing suggestions relevant to the destination
    - Tips for enjoying the destination
    
    Base your plans on the research provided, while considering the user's preferences, budget constraints, and time frame.
    """
)

thinking_agent = Agent(
    name="Thinking Agent",
    handoff_description="Reflective agent that explains reasoning and considerations",
    instructions="""You are a reflective thinking agent. Your purpose is to make the decision-making process transparent.
    
    For any given travel query or planning situation:
    1. Analyze what information is known and unknown
    2. Consider multiple perspectives and approaches
    3. Evaluate tradeoffs between different options
    4. Identify assumptions being made
    5. Ask critical questions that could change recommendations
    6. Provide reasoning for final recommendations
    
    Your goal is to show the thought process behind travel recommendations and plans, not to make the recommendations yourself.
    """
)

# Create the triage agent that will coordinate
triage_agent = Agent(
    name="Travel Assistant",
    instructions="""You are a helpful travel assistant coordinating the planning process. 
    
    Based on user queries:
    1. If they need information about destinations, hand off to the Research Agent
    2. If they need a detailed itinerary or plan, hand off to the Planning Agent
    3. If they want to understand the reasoning process, hand off to the Thinking Agent
    
    You should first understand the user's query, then determine the most appropriate specialist agent.
    When receiving information back from specialist agents, integrate it into your response to the user.
    """,
    handoffs=[research_agent, planning_agent, thinking_agent]
)

# Store conversation history
conversation_history = []

# Function to run agent asynchronously
async def run_agent(query):
    # Create a context to track the flow
    context = {}
    
    # First run the triage agent
    print(f"Running Triage Agent with query: {query}")
    triage_result = await Runner.run(triage_agent, query)
    
    # Check if there's a handoff
    handoff_agent = None
    triage_response = triage_result.final_output
    conversation_flow = [{
        "agent": "Triage Agent",
        "response": triage_response
    }]

    # Run a simplified handoff detection
    handoff_detected = False
    
    # Check for research agent handoff
    if "Research Agent" in triage_response or "information about destinations" in triage_response.lower():
        print("Handing off to Research Agent")
        handoff_agent = research_agent
        handoff_detected = True
        
    # Check for planning agent handoff
    elif "Planning Agent" in triage_response or "itinerary" in triage_response.lower() or "plan" in triage_response.lower():
        print("Handing off to Planning Agent")
        handoff_agent = planning_agent
        handoff_detected = True
        
    # Check for thinking agent handoff
    elif "Thinking Agent" in triage_response or "reasoning" in triage_response.lower() or "think through" in triage_response.lower():
        print("Handing off to Thinking Agent")
        handoff_agent = thinking_agent
        handoff_detected = True
    
    # If handoff is detected, run the specialist agent
    if handoff_detected and handoff_agent:
        specialist_result = await Runner.run(handoff_agent, query)
        specialist_response = specialist_result.final_output
        
        # Add specialist agent to the flow
        conversation_flow.append({
            "agent": handoff_agent.name,
            "response": specialist_response
        })
        
        # Final integration by triage agent
        integration_prompt = f"""Based on the user query: {query}

The specialist {handoff_agent.name} provided this response: {specialist_response}

Please integrate this information into a final coherent response."""
        
        integration_result = await Runner.run(triage_agent, integration_prompt)
        integration_response = integration_result.final_output
        
        # Add integration to the flow
        conversation_flow.append({
            "agent": "Triage Agent (Integration)",
            "response": integration_response
        })
        
        return {
            "final_response": integration_response,
            "conversation_flow": conversation_flow
        }
    
    # If no handoff, just return the triage response
    return {
        "final_response": triage_response,
        "conversation_flow": conversation_flow
    }

# Helper function to run async code from sync functions
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

# Routes
@app.route('/')
def index():
    return render_template('index.html', conversation=conversation_history)

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Add user message to conversation history
    conversation_history.append({
        'role': 'user',
        'content': query
    })
    
    # Get response from agent
    result = run_async(run_agent(query))
    
    # Get the final response and conversation flow
    final_response = result["final_response"]
    conversation_flow = result["conversation_flow"]
    
    # Add agent response to conversation history
    conversation_history.append({
        'role': 'assistant',
        'content': final_response,
        'flow': conversation_flow
    })
    
    return jsonify({
        'response': final_response,
        'conversation': conversation_history,
        'flow': conversation_flow
    })

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'Conversation reset successfully'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)
