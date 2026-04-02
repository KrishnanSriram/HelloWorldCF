
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    BingGroundingTool,
    MessageRole,
    Agent,
    AgentThread,
)

load_dotenv()


def create_project_client() -> AIProjectClient:
    client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    print("✅ Connected to Foundry Project")
    return client


def create_bing_tool(client: AIProjectClient) -> BingGroundingTool:
    # Step 1: look up the connection by name from your project
    bing_connection = client.connections.get(
        name=os.environ["BING_CONNECTION_NAME"]
    )
    print(f"✅ Bing connection found: {bing_connection.id}")

    # Step 2: wrap it as a tool using the connection ID
    return BingGroundingTool(connection_id=bing_connection.id)


def create_bing_agent(
    client: AIProjectClient,
    bing_tool: BingGroundingTool,
) -> Agent:
    agent = client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="news-reporter",
        instructions="""
            You are a helpful news and research assistant with access to
            real-time web search via Bing.
            Always search for current information before answering.
            Include source citations in your response where available.
            Keep responses concise and well structured.
        """,
        tools=bing_tool.definitions,    # ← same pattern as CodeInterpreterTool
    )
    print(f"✅ Agent created: {agent.id}")
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

    # ✅ create_and_process works here because Bing runs server-side
    run = client.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    if run.status == "completed":
        messages = client.agents.messages.list(thread_id=thread.id)
        for msg in messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        print(f"\n🤖 Agent: {block.text.value}")
                break
    else:
        print(f"❌ Run failed: {run.status}")
        if run.last_error:
            print(f"   Error: {run.last_error.message}")


def main():
    client     = create_project_client()
    bing_tool  = create_bing_tool(client)
    agent      = create_bing_agent(client, bing_tool)
    thread     = client.agents.threads.create()
    print(f"✅ Thread created: {thread.id}")

    ask_agent(client, thread, agent, "What are the top tech news stories today?")
    ask_agent(client, thread, agent, "What is the latest version of Python?")
    ask_agent(client, thread, agent, "Summarise the most recent Azure AI announcements.")

    client.agents.delete_agent(agent.id)
    print("\n🧹 Agent cleaned up")


if __name__ == "__main__":
    main()