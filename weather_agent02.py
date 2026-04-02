# weather_agent02.py
# Dependencies: azure-ai-projects==1.0.0, azure-ai-agents==1.1.0
# Uses OpenWeatherMap free tier — sign up at https://openweathermap.org/api
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


def get_current_weather(city: str, unit: str = "imperial") -> str:
    """
    Fetches current weather for a given city.
    Falls back to mock data if no API key is set.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if api_key:
        # Real OpenWeatherMap API call
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
                "feels_like":  f"{data['main']['feels_like']}°{'F' if unit == 'imperial' else 'C'}",
                "condition":   data["weather"][0]["description"],
                "humidity":    f"{data['main']['humidity']}%",
                "wind_speed":  f"{data['wind']['speed']} mph",
            })
        except requests.RequestException as e:
            return json.dumps({"error": f"API call failed: {str(e)}"})
    else:
        # 🌧️ Mock data — because it's always raining in London
        mock_data = {
            "london":  {"temperature": "55°F", "condition": "light rain",   "humidity": "82%", "wind_speed": "12 mph"},
            "chicago": {"temperature": "28°F", "condition": "heavy snow",   "humidity": "91%", "wind_speed": "25 mph"},
            "dublin":  {"temperature": "48°F", "condition": "cloudy",       "humidity": "78%", "wind_speed": "15 mph"},
            "sydney":  {"temperature": "77°F", "condition": "sunny",        "humidity": "55%", "wind_speed": "8 mph"},
        }
        city_lower = city.lower()
        weather = mock_data.get(city_lower, {
            "temperature": "72°F",
            "condition":   "clear skies",
            "humidity":    "60%",
            "wind_speed":  "5 mph",
        })
        return json.dumps({"city": city, **weather, "source": "mock"})


def get_weather_forecast(city: str, days: int = 3) -> str:
    """
    Returns a simple multi-day forecast for a city.
    Days must be between 1 and 5.
    """
    days = max(1, min(days, 5))  # clamp to 1–5

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if api_key:
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?q={city}&cnt={days * 8}&appid={api_key}&units=imperial"
        )
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            # Grab one reading per day (every 8th entry = ~24hrs apart)
            forecast = []
            for i in range(0, min(days * 8, len(data["list"])), 8):
                entry = data["list"][i]
                forecast.append({
                    "date":        entry["dt_txt"].split(" ")[0],
                    "temperature": f"{entry['main']['temp']}°F",
                    "condition":   entry["weather"][0]["description"],
                })
            return json.dumps({"city": city, "forecast": forecast})
        except requests.RequestException as e:
            return json.dumps({"error": f"API call failed: {str(e)}"})
    else:
        # Mock forecast
        mock_forecast = [
            {"date": "2026-03-27", "temperature": "57°F", "condition": "partly cloudy"},
            {"date": "2026-03-28", "temperature": "61°F", "condition": "sunny"},
            {"date": "2026-03-29", "temperature": "53°F", "condition": "light rain"},
            {"date": "2026-03-30", "temperature": "49°F", "condition": "overcast"},
            {"date": "2026-03-31", "temperature": "55°F", "condition": "sunny"},
        ]
        return json.dumps({
            "city":     city,
            "forecast": mock_forecast[:days],
            "source":   "mock",
        })

def build_toolset() -> ToolSet:
    weather_tool_set = ToolSet()
    weather_tool_set.add(FunctionTool(functions={get_current_weather, get_weather_forecast}))
    return weather_tool_set


def create_project_client() -> AIProjectClient:
    client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    print("✅ Connected to Foundry Project")
    return client


def create_weather_agent(client: AIProjectClient, toolset: ToolSet) -> Agent:

    agent = client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="weather-reporter",
        instructions="""
            You are a friendly weather reporting assistant.
            You have two tools:
            - get_current_weather: use when the user asks about current conditions
            - get_weather_forecast: use when the user asks about upcoming days
            Always mention the city name in your response.
            If the data source is 'mock', let the user know it's simulated data.
            Keep responses concise and conversational.
        """,
        toolset=toolset,
    )
    print(f"✅ Weather agent created: {agent.id}")
    return agent


def ask_agent(
    client: AIProjectClient,
    thread: AgentThread,
    agent: Agent,
    question: str,
) -> None:
    print(f"\n👤 You: {question}")

    client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=question,
    )

    # Start the run (don't use create_and_process)
    run = client.agents.runs.create(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    # Poll until terminal state
    while run.status in ("queued", "in_progress", "requires_action"):
        time.sleep(1)
        run = client.agents.runs.get(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for tc in tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                # Dispatch to the right local function
                if func_name == "get_current_weather":
                    result = get_current_weather(**func_args)
                elif func_name == "get_weather_forecast":
                    result = get_weather_forecast(**func_args)
                else:
                    result = json.dumps({"error": f"Unknown function: {func_name}"})

                tool_outputs.append({
                    "tool_call_id": tc.id,
                    "output": result,
                })

            # Submit results back to the run
            client.agents.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
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
    client = create_project_client()
    weather_tool_set = build_toolset()
    agent = create_weather_agent(client, weather_tool_set)
    thread = client.agents.threads.create()

    print(f"✅ Thread created: {thread.id}")

    ask_agent(client, thread, agent, "What's the weather like in London right now?")
    ask_agent(client, thread, agent, "Give me a 3-day forecast for Sydney.")
    ask_agent(client, thread, agent, "Should I pack an umbrella for Chicago tomorrow?")

    client.agents.delete_agent(agent.id)
    print("\n🧹 Agent cleaned up")


if __name__ == "__main__":
    main()