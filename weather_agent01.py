# weather_agent02.py
# Dependencies: azure-ai-projects==1.0.0, azure-ai-agents==1.1.0
import os
import json
import requests
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    FunctionTool,
    ToolSet,
    Agent,
    AgentThread,
    MessageRole,
)
import time

load_dotenv()


# ─────────────────────────────────────────────────────────────
# THE TOOL FUNCTION
#
# This is plain Python — no Azure imports, no SDK awareness.
# The agent never runs this directly. It tells YOU to run it,
# you run it, and you hand the result back. The SDK's
# create_and_process() handles that handoff automatically.
#
# Three things matter here for the SDK to auto-generate
# the tool schema:
#   1. Type hints on every parameter  (city: str, unit: str)
#   2. A clear docstring               (tells the agent WHAT this does)
#   3. Return a STRING                 (always — JSON string is best)
# ─────────────────────────────────────────────────────────────
def get_current_weather(city: str, unit: str = "imperial") -> str:
    """
    Get the current weather for a given city.
    Use this when the user asks about current weather conditions.

    Parameters:
        city: The name of the city, e.g. 'London' or 'Chicago'
        unit: Temperature unit — 'imperial' for Fahrenheit, 'metric' for Celsius
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if api_key:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&units={unit}&appid={api_key}"
        )
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return json.dumps({
                "city":        data["name"],
                "temperature": f"{data['main']['temp']}°{'F' if unit == 'imperial' else 'C'}",
                "condition":   data["weather"][0]["description"],
                "humidity":    f"{data['main']['humidity']}%",
                "wind_speed":  f"{data['wind']['speed']} mph",
            })
        except requests.RequestException as e:
            return json.dumps({"error": str(e)})
    else:
        # Mock fallback — no API key needed to develop
        mock = {
            "london":  {"temperature": "55°F", "condition": "light rain",  "humidity": "82%"},
            "chicago": {"temperature": "28°F", "condition": "heavy snow",  "humidity": "91%"},
            "dublin":  {"temperature": "48°F", "condition": "cloudy",      "humidity": "78%"},
            "sydney":  {"temperature": "77°F", "condition": "sunny",       "humidity": "55%"},
        }
        data = mock.get(city.lower(), {
            "temperature": "72°F", "condition": "clear skies", "humidity": "60%"
        })
        return json.dumps({"city": city, **data, "source": "mock"})


# ─────────────────────────────────────────────────────────────
# WRAP THE FUNCTION AS A TOOL
#
# FunctionTool takes a SET of functions.
# The SDK reads the type hints + docstring and auto-generates
# the JSON schema that tells the agent:
#   - what the function is called
#   - what it does (from the docstring)
#   - what arguments to pass (from type hints)
#   - which arguments are required vs optional
#
# You never write the schema manually — Python does it for you.
# ─────────────────────────────────────────────────────────────
def build_toolset() -> ToolSet:
    toolset = ToolSet()
    toolset.add(FunctionTool(functions={get_current_weather}))
    return toolset

# ─────────────────────────────────────────────────────────────
# PROJECT CLIENT
# Same as the research agent — connects to your Foundry Project
# ─────────────────────────────────────────────────────────────
def create_project_client() -> AIProjectClient:
    client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    print("✅ Connected to Foundry Project")
    return client


# ─────────────────────────────────────────────────────────────
# CREATE AGENT
#
# The ToolSet bundles one or more tools together.
# The agent's instructions tell it WHEN to use the tool
# and HOW to present results. Good instructions here
# are just as important as good tool descriptions.
# ─────────────────────────────────────────────────────────────
def create_weather_agent(client: AIProjectClient, toolset: ToolSet) -> Agent:
    agent = client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="weather-reporter",
        instructions="""
            You are a friendly weather assistant.
            When the user asks about current weather, use get_current_weather.
            Always mention the city name in your response.
            If the response contains 'source: mock', tell the user
            the data is simulated.
            Keep responses short and conversational.
        """,
        toolset=toolset,
    )
    print(f"✅ Agent created: {agent.id}")
    return agent


# ─────────────────────────────────────────────────────────────
# ASK THE AGENT A QUESTION
#
# This is the core agentic loop — what happens inside
# create_and_process() on each call:
#
#   1. Your message lands in the thread
#   2. Agent reads the message + all tools available
#   3. Agent decides: do I need a tool?
#      → YES: agent emits a "tool call" (city="London")
#             SDK detects it, runs your function
#             result goes back into the thread
#             agent reads result, formulates response
#      → NO:  agent responds directly from its own knowledge
#   4. Final response appears in the thread
#
# The thread keeps ALL messages — so the agent remembers
# everything said earlier in the conversation.
# ─────────────────────────────────────────────────────────────
def ask_agent(
    client: AIProjectClient,
    thread: AgentThread,
    agent: Agent,
    question: str,
    toolset: ToolSet,
) -> None:
    print(f"\n👤 You: {question}")

    client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=question,
    )

    run = client.agents.runs.create(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    while run.status in ("queued", "in_progress", "requires_action"):
        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            print(f"🔧 Agent calling: {[tc.function.name for tc in tool_calls]}")
            tool_outputs = toolset.execute_tool_calls(tool_calls)
            client.agents.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
        else:
            time.sleep(1)

        run = client.agents.runs.get(
            thread_id=thread.id,
            run_id=run.id,
        )

    if run.status == "completed":
        messages = client.agents.messages.list(thread_id=thread.id)
        for msg in messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        print(f"🤖 Agent: {block.text.value}")
                break
    else:
        print(f"❌ Run failed: {run.status}")
        if run.last_error:
            print(f"   Error: {run.last_error.message}")


def main():
    client   = create_project_client()
    toolset  = build_toolset()              # ✅ build toolset separately so we can pass it around
    agent    = create_weather_agent(client, toolset)
    thread   = client.agents.threads.create()
    print(f"✅ Thread created: {thread.id}")

    ask_agent(client, thread, agent, "What's the weather in London?", toolset)
    ask_agent(client, thread, agent, "How about Chicago?", toolset)
    ask_agent(client, thread, agent, "Which one is warmer?", toolset)

    client.agents.delete_agent(agent.id)
    print("\n🧹 Agent cleaned up")


if __name__ == "__main__":
    main()