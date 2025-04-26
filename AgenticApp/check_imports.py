import agents

# Print available modules/attributes in the agents package
print(dir(agents))

# Try to find TraceEvent
try:
    from agents.tracing import TraceEvent
    print("Successfully imported TraceEvent from agents.tracing")
except ImportError as e:
    print(f"Error importing TraceEvent from agents.tracing: {e}")

# Try alternative imports
try:
    from agents import TraceEvent
    print("Successfully imported TraceEvent directly from agents")
except ImportError as e:
    print(f"Error importing TraceEvent directly from agents: {e}")

# Check guardrail module
try:
    from agents.guardrail import InputGuardrail, GuardrailFunctionOutput
    print("Successfully imported InputGuardrail and GuardrailFunctionOutput from agents.guardrail")
except ImportError as e:
    print(f"Error importing from agents.guardrail: {e}")
