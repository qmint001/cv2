import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Import the basic Agent and Runner classes
from agents import Agent, Runner

# Create a simple agent
agent = Agent(
    name="Simple Agent",
    instructions="You are a helpful assistant."
)

# Run the agent with a simple prompt
import asyncio

async def main():
    result = await Runner.run(agent, "Hello, how are you?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
