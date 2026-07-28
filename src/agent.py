"""LangGraph orchestration for the local paper reading agent."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .llm import LLMConfig, create_chat_model, get_default_llm_config
from .prompts import SYSTEM_PROMPT


class AgentState(TypedDict):
    """Messages shared between the reasoning and tool nodes."""

    messages: Annotated[list, add_messages]


def build_tools(llm_config: LLMConfig):
    """Bind paper tools to one immutable model configuration."""

    @tool
    def search_papers(query: str, max_results: int = 5) -> str:
        """Search arXiv for papers on a topic. Returns formatted results with IDs."""
        from .tools import search_papers as run

        return run(query, max_results)

    @tool
    def download_paper(arxiv_id: str) -> str:
        """Download a paper PDF from arXiv by ID (e.g. 1706.03762)."""
        from .tools import download_paper as run

        return run(arxiv_id)

    @tool
    def summarize_papers(arxiv_ids: str) -> str:
        """Summarize papers by comma-separated arXiv IDs."""
        from .tools import summarize_papers as run

        return run(arxiv_ids, llm_config=llm_config)

    @tool
    def deep_read(arxiv_id: str) -> str:
        """Deep structured analysis of a paper by arXiv ID."""
        from .tools import deep_read as run

        return run(arxiv_id, llm_config=llm_config)

    @tool
    def chat_with_paper(arxiv_id: str, question: str) -> str:
        """Ask a question about a downloaded paper."""
        from .tools import chat_with_paper as run

        return run(arxiv_id, question, llm_config=llm_config)

    @tool
    def index_library() -> str:
        """Index all downloaded PDFs in papers/ into the local vector library."""
        from .tools import index_library as run

        return run()

    @tool
    def search_library(query: str, n_results: int = 5) -> str:
        """Semantic search across locally indexed papers."""
        from .tools import search_library as run

        return run(query, n_results)

    @tool
    def library_stats() -> str:
        """Check how many papers are indexed in the local library."""
        from .tools import library_stats as run

        return run()

    return [
        search_papers,
        download_paper,
        summarize_papers,
        deep_read,
        chat_with_paper,
        index_library,
        search_library,
        library_stats,
    ]


def create_agent(llm_config: LLMConfig | None = None):
    """Build a compiled graph whose reasoning and AI tools share one model."""

    active_config = (llm_config or get_default_llm_config()).require_valid()
    active_tools = build_tools(active_config)
    llm_with_tools = create_chat_model(active_config).bind_tools(active_tools)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
        )
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(active_tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


def get_agent(llm_config: LLMConfig | None = None):
    """Create an agent for the requested model configuration."""

    return create_agent(llm_config)
