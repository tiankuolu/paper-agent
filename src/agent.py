"""LangGraph Agent for paper reading.
构建一个 LLM + Tools 的 Agent 图（Graph）：
  agent_node（LLM 推理） ←→ tools_node（工具执行）

这是整个系统的"大脑"——它接收用户消息，决定调用哪个工具，
拿到工具结果后继续推理，直到能给出最终回复。"""

from typing import TypedDict, Annotated  # 类型注解，定义图的 State 结构
from langgraph.graph import StateGraph  # LangGraph 的状态图，节点+边的有向图
from langgraph.graph.message import add_messages  # 消息列表的合并策略：追加而不是覆盖
from langgraph.prebuilt import ToolNode, tools_condition  # ToolNode: 执行工具；tools_condition: 判断 LLM 是否要调工具
from langgraph.checkpoint.memory import MemorySaver  # 内存中的检查点存储，实现多轮对话记忆
from langchain_openai import ChatOpenAI  # LangChain 的 OpenAI 兼容 LLM 封装，连接 DeepSeek
from langchain_core.tools import tool  # @tool 装饰器，把 Python 函数转为 LLM 可调用的 tool schema

from .prompts import SYSTEM_PROMPT  # 系统提示词，定义 agent 角色和工作流程
import os  # 读取环境变量
from dotenv import load_dotenv  # 加载 .env 配置
load_dotenv()  # 执行加载，把 DEEPSEEK_API_KEY 等写入 os.environ

# ============================================================
# Agent 状态定义
# State 是图中各节点共享的数据结构，在节点间流转
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史，add_messages 表示新消息会追加到列表末尾，不会覆盖旧消息

# ============================================================
# 工具定义：每个 @tool 装饰的函数 = agent 的一个"技能"
# docstring 会自动转为 tool description，LLM 据此判断何时调用
# 函数体做延迟导入，避免启动时的循环依赖
# ============================================================

@tool
def search_papers(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers on a topic. Returns formatted results with IDs."""
    from .tools import search_papers as _s  # 延迟导入，运行时才加载 tools 模块
    return _s(query, max_results)  # 委托给 tools 模块的实际实现

@tool
def download_paper(arxiv_id: str) -> str:
    """Download a paper PDF from arXiv by ID (e.g. 1706.03762)."""
    from .tools import download_paper as _d
    return _d(arxiv_id)

@tool
def summarize_papers(arxiv_ids: str) -> str:
    """Summarize papers by comma-separated arXiv IDs."""
    from .tools import summarize_papers as _sm
    return _sm(arxiv_ids)

@tool
def deep_read(arxiv_id: str) -> str:
    """Deep structured analysis of a paper by arXiv ID."""
    from .tools import deep_read as _dr
    return _dr(arxiv_id)

@tool
def chat_with_paper(arxiv_id: str, question: str) -> str:
    """Ask a question about a downloaded paper."""
    from .tools import chat_with_paper as _c
    return _c(arxiv_id, question)

@tool
def index_library() -> str:
    """Index all downloaded PDFs in papers/ into the local vector library for semantic search."""
    from .tools import index_library as _il
    return _il()

@tool
def search_library(query: str, n_results: int = 5) -> str:
    """Semantic search across locally indexed papers. Finds the most relevant passages."""
    from .tools import search_library as _sl
    return _sl(query, n_results)

@tool
def library_stats() -> str:
    """Check how many papers are indexed in the local library."""
    from .tools import library_stats as _ls
    return _ls()

# 所有工具的列表，会绑定到 LLM，LLM 根据 docstring 自动选择调用
ALL_TOOLS = [search_papers, download_paper, summarize_papers, deep_read, chat_with_paper, index_library, search_library, library_stats]

# ============================================================
# Agent 图构建
# 结构：START → agent_node ⇄ tools_node（循环直到 LLM 不再调工具）
# ============================================================
def create_agent():
    """构建并编译 LangGraph agent 图。返回可调用的 compiled graph。"""

    # ---- LLM 配置 ----
    llm = ChatOpenAI(
        model="deepseek-v4-pro",  # DeepSeek 模型名
        temperature=0.3,  # 低温度 = 输出更确定，适合工具调用场景
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # API Key
        openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # DeepSeek 兼容 OpenAI 格式
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)  # 把工具 schema 绑定到 LLM，LLM 会在需要时返回 tool_call

    # ---- agent_node：推理节点 ----
    # 每次执行：发送 system prompt + 全部历史消息 → LLM → 返回 AI 消息（可能是 tool_call 或最终回复）
    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}]  # 系统提示词放在最前面
            + state["messages"]  # 然后拼接历史消息（用户消息 + 之前的 AI 回复 + 工具结果）
        )
        return {"messages": [response]}  # 返回的消息会被 add_messages 追加到 state.messages

    # ---- 构建图 ----
    graph = StateGraph(AgentState)  # 创建状态图，指定状态类型
    graph.add_node("agent", agent_node)  # 添加 agent 节点
    graph.add_node("tools", ToolNode(ALL_TOOLS))  # 添加工具节点，ToolNode 自动根据 tool_call 执行对应函数
    graph.set_entry_point("agent")  # 入口是 agent 节点
    graph.add_conditional_edges(  # agent 节点之后的條件边：
        "agent",  # 从 agent 节点出发
        tools_condition,  # 判断条件：如果 LLM 输出了 tool_call → 去 tools；否则 → END
    )
    graph.add_edge("tools", "agent")  # tools 节点执行完后，无条件回到 agent 节点继续推理

    # ---- 记忆 ----
    memory = MemorySaver()  # 内存中的检查点存储，服务重启后会丢失
    return graph.compile(checkpointer=memory)  # 编译图，注入记忆组件

# ============================================================
# 单例模式：全局只创建一个 agent 实例，避免重复编译
# ============================================================
_agent = None  # 模块级缓存

def get_agent():
    """获取全局唯一的 agent 实例（延迟初始化）。"""
    global _agent
    if _agent is None:  # 第一次调用才创建
        _agent = create_agent()
    return _agent
