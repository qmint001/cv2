import os
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Import just the core functionality we need
from agents import Agent, Runner

# Define our data structures
class Destination(BaseModel):
    name: str
    description: str
    attractions: List[str] = Field(default_factory=list)
    accommodations: List[str] = Field(default_factory=list)

class TravelPlan(BaseModel):
    destination: str
    duration: str
    activities: List[Dict[str, str]] = Field(default_factory=list)
    accommodations: str
    transportation: str
    estimated_budget: str
    notes: str = ""

class ThinkingOutput(BaseModel):
    reasoning: str
    considerations: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    conclusion: str

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
    """,
    output_type=TravelPlan
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
    """,
    output_type=ThinkingOutput
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

# Function to trace the conversation flow
def print_conversation_flow(result):
    print("\n=== CONVERSATION FLOW ===\n")
    
    # Just print that the query was processed by the agent
    print(f"Flow: User → Triage Agent")
    
    # For version 0.0.9, we can't access the detailed flow information
    # so we'll just acknowledge that a response was generated
    print("Response generated successfully")

# Main function to run our travel planning assistant
async def main():
    # Example user queries
    queries = [
        "I'm thinking about visiting Tokyo for a week in April. What should I know?",
        "Can you create a 3-day itinerary for San Francisco on a budget?",
        "I'm trying to decide between Bali and Thailand for a 2-week trip. Can you explain how you'd think through this decision?"
    ]
    
    for query in queries:
        print(f"\n\nUSER QUERY: {query}\n")
        
        # Run the triage agent with the query
        result = await Runner.run(triage_agent, query)
        
        # Print the final output
        print("\nFINAL RESPONSE:")
        print(result.final_output)
        
        # Print the conversation flow
        print_conversation_flow(result)
        
        print("\n" + "-"*80)

if __name__ == "__main__":
    asyncio.run(main())
