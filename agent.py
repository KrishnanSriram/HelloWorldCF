# agent.py — Simple Research Assistant using azure-ai-projects v2 beta
# agent.py — azure-ai-projects 1.0.0 stable + azure-ai-agents 1.1.0
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

# ✅ On 1.0.0 stable, ALL of these live in azure.ai.agents.models
#    (azure-ai-agents 1.x is the companion package that ships the agent types)
from azure.ai.agents.models import (
    CodeInterpreterTool,
    MessageRole,
    Agent,
    AgentThread,
    ThreadRun,
)

load_dotenv()


def create_project_client() -> AIProjectClient:
    client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    print("✅ Connected to Foundry Project")
    return client


def create_agent(client: AIProjectClient) -> Agent:  # ✅ real SDK type
    agent = client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="research-assistant",
        instructions="""
            You are a helpful research assistant.
            When asked questions, provide clear, structured answers.
            Use code interpreter when you need to do calculations or data analysis.
            Always cite your reasoning step by step.
        """,
        tools=CodeInterpreterTool().definitions,
    )
    print(f"✅ Agent created: {agent.id} — '{agent.name}'")
    return agent


def create_thread(client: AIProjectClient) -> AgentThread:  # ✅ real SDK type
    thread = client.agents.threads.create()
    print(f"✅ Thread created: {thread.id}")
    return thread


def client_questions() -> str:
    user_question = (
        "Explain the top 3 benefits of using Azure AI Foundry for building "
        "agentic solutions, and calculate how many hours are in a 90-day sprint."
    )
    return user_question


def create_message_for_agent(
    client: AIProjectClient,
    thread: AgentThread,
    user_question: str,
) -> None:
    client.agents.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=user_question,
    )
    print(f"✅ Message sent: '{user_question[:60]}...'")


def run_agent(
    client: AIProjectClient,
    thread: AgentThread,
    agent: Agent,
) -> ThreadRun:  # ✅ real SDK type
    print("\n⏳ Running agent (this may take a moment)...\n")

    run = client.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    print(f"✅ Run completed — Status: {run.status}")
    return run  # ✅ now actually returned


def process_agent_response(
    client: AIProjectClient,
    thread: AgentThread,
    agent: Agent,
    run: ThreadRun,
) -> None:
    if run.status == "completed":
        messages = client.agents.messages.list(thread_id=thread.id)

        print("\n" + "=" * 60)
        print("🤖 AGENT RESPONSE")
        print("=" * 60)

        for msg in messages:
            if msg.role == "assistant":          # ✅ plain string, not MessageRole.ASSISTANT
                for content_block in msg.content:
                    if hasattr(content_block, "text"):
                        print(content_block.text.value)
                break
    else:
        print(f"❌ Run failed with status: {run.status}")
        if run.last_error:
            print(f"   Error: {run.last_error.message}")

    client.agents.delete_agent(agent.id)
    print("\n🧹 Agent cleaned up")


def main():
    client = create_project_client()
    agent = create_agent(client)
    thread = create_thread(client)
    user_question = client_questions()
    create_message_for_agent(client, thread, user_question)
    run = run_agent(client, thread, agent)      # ✅ now captures the return value
    process_agent_response(client, thread, agent, run)


if __name__ == "__main__":
    main()