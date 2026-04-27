import os
import sys
import json
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_PROJECT"] = "social-research-agent"

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    topic: str
    research: str
    result: dict

class SocialPost(BaseModel):
    title: str = Field(description="A fancy, attention-grabbing title (max 80 chars)")
    summary: str = Field(description="A 2-3 line summary suitable for a social media post")

LLM = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
SEARCH = TavilySearch(max_results=4)

def research_node(state: AgentState) -> dict:
    print(f"🔍 Researching: {state['topic']}")
    search_result = SEARCH.invoke({"query": state["topic"]})

    items = search_result.get("results", []) if isinstance(search_result, dict) else []
    research_text = "\n\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')}" for r in items
    )
    sources = [r.get("url") for r in items if r.get("url")]

    return {"research": research_text, "result": {"sources": sources}}

def summarize_node(state: AgentState) -> dict:
    print("✍️  Generating title and summary...")

    structured_llm = LLM.with_structured_output(SocialPost)

    prompt = (
        f"You are a social media copywriter. Based on the research below about "
        f"'{state['topic']}', write:\n"
        f"1. A fancy, attention-grabbing title (max 80 chars)\n"
        f"2. A 2-3 line summary perfect for a social media post (LinkedIn/Twitter)\n\n"
        f"RESEARCH:\n{state['research']}"
    )

    post: SocialPost = structured_llm.invoke(prompt)

    final = {
        "topic": state["topic"],
        "title": post.title,
        "summary": post.summary,
        "sources": state["result"].get("sources", []),
    }
    return {"result": final}

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("research", research_node)
    g.add_node("summarize", summarize_node)
    g.add_edge(START, "research")
    g.add_edge("research", "summarize")
    g.add_edge("summarize", END)
    return g.compile()

def run(topic: str) -> dict:
    graph = build_graph()
    final_state = graph.invoke({"topic": topic})
    return final_state["result"]

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "Latest developments in agentic AI"
    output = run(topic)
    print("\n" + "=" * 60)
    print("📦 FINAL JSON OUTPUT")
    print("=" * 60)
    print(json.dumps(output, indent=2, ensure_ascii=False))