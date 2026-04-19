# filesearch_agent.py
# Dependencies: azure-ai-projects==1.0.0, azure-ai-agents==1.1.0
#
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    FileSearchTool,
    MessageRole,
    Agent,
    AgentThread,
    VectorStore,
)
load_dotenv()


def create_project_client() -> AIProjectClient:
    client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    print("✅ Connected to Foundry Project")
    return client


def upload_file(client: AIProjectClient, file_path: str):
    with open(file_path, "rb") as f:
        uploaded_file = client.agents.files.upload(
            file=f,
            purpose="assistants",       # tells Azure this file is for agent search
        )
    print(f"✅ File uploaded: {uploaded_file.id} - '{file_path}'")
    return uploaded_file


def create_vector_store(
    client: AIProjectClient,
    file_id: str,
    name: str = "policy-store",
) -> VectorStore:
    vector_store = client.agents.vector_stores.create_and_poll(
        file_ids=[file_id],     # attach our uploaded file
        name=name,
    )
    print(f"✅ Vector store ready: {vector_store.id} - status: {vector_store.status}")
    return vector_store


def create_file_search_tool(vector_store: VectorStore) -> FileSearchTool:
    return FileSearchTool(vector_store_ids=[vector_store.id])


def create_search_agent(
    client: AIProjectClient,
    file_search_tool: FileSearchTool,
) -> Agent:
    agent = client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="policy-assistant",
        instructions="""
            You are a helpful HR assistant.
            Answer questions using only the information in the uploaded
            policy documents. Always cite the specific policy section
            your answer comes from.
            If the answer is not in the documents, say so clearly -
            do not make up information.
        """,
        tools=file_search_tool.definitions,
        tool_resources=file_search_tool.resources,  # ← links the vector store to the agent
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


def cleanup(
    client: AIProjectClient,
    agent: Agent,
    vector_store: VectorStore,
    file_id: str,
) -> None:
    client.agents.delete_agent(agent.id)
    print("\n🧹 Agent deleted")
    client.agents.vector_stores.delete(vector_store.id)
    print("🧹 Vector store deleted")
    client.agents.files.delete(file_id)
    print("🧹 File deleted")

def main():
    client       = create_project_client()
    # Upload and index the file
    uploaded     = upload_file(client, "company_policy.txt")
    vector_store = create_vector_store(client, uploaded.id)
    tool         = create_file_search_tool(vector_store)
    # Create agent and thread
    agent        = create_search_agent(client, tool)
    thread       = client.agents.threads.create()
    print(f"✅ Thread created: {thread.id}")
    # Ask questions grounded in the document
    ask_agent(client, thread, agent, "How many vacation days do full-time employees get?")
    ask_agent(client, thread, agent, "What is the daily meal allowance for business travel?")
    ask_agent(client, thread, agent, "Can I work from home every day?")
    ask_agent(client, thread, agent, "When do salary adjustments take effect after a review?")
    # Clean up all Azure resources
    cleanup(client, agent, vector_store, uploaded.id)

if __name__ == "__main__":
    main()