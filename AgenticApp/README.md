# Travel Planning Assistant with OpenAI Agentic Framework

This application demonstrates the use of the OpenAI Agents SDK to create a travel planning assistant with multiple specialized agents and a thinking agent that makes the reasoning process transparent.

## Architecture

The application consists of four agents:

1. **Triage Agent**: The main coordinator that receives user queries and delegates to specialized agents
2. **Research Agent**: Specializes in gathering information about travel destinations
3. **Planning Agent**: Creates detailed travel itineraries based on research
4. **Thinking Agent**: Makes the reasoning process transparent by explaining considerations and tradeoffs

## Conversation Flow

The application demonstrates agent handoffs and includes a visualization of the conversation flow, showing:
- Agent activations
- Handoffs between agents
- Tool calls
- Final outputs

## Setup

1. Install the OpenAI Agents SDK:
   ```
   pip install openai-agents
   ```

2. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=sk-your-api-key
   ```
   Or uncomment and set the key in the code:
   ```python
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   ```

3. Run the application:
   ```
   python travel_planning_app.py
   ```

## Key Features

- **Multi-agent orchestration**: Demonstrates how to use handoffs to delegate between specialized agents
- **Structured output**: Uses Pydantic models to define structured output schemas
- **Transparent reasoning**: Includes a thinking agent to show the reasoning process
- **Conversation flow visualization**: Traces the conversation flow between agents

## Extending the Application

You can extend this application by:

1. Adding more specialized agents (e.g., Budget Agent, Cultural Insights Agent)
2. Implementing tools for real-time data (e.g., flight searches, weather forecasts)
3. Adding guardrails for input validation
4. Improving the conversation flow visualization
