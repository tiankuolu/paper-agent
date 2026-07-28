"""Streamlit UI for Paper Reading Agent.
本地 Web 界面，提供对话式论文助手 + 论文库管理。
运行方式: streamlit run app.py
"""

import streamlit as st
import uuid
from dotenv import load_dotenv
load_dotenv()

import os

# ---- 页面配置 ----
st.set_page_config(
    page_title="📚 Paper Reading Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 延迟导入 agent（带错误提示） ----
@st.cache_resource
def load_agent():
    """加载 Agent 单例，缓存避免重复初始化。"""
    from src.agent import get_agent
    return get_agent()

# ---- 初始化 session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ---- 侧边栏 ----
with st.sidebar:
    st.title("📚 Paper Agent")
    st.caption("LangGraph + ReAct + RAG")

    # API 状态
    st.divider()
    st.subheader("🔑 API 状态")
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key:
        st.success(f"DeepSeek: {key[:10]}***")
    else:
        st.error("未配置 API Key")
        st.code("echo DEEPSEEK_API_KEY=sk-xxx > .env", language="bash")

    # 论文库
    st.divider()
    st.subheader("📦 论文库")

    # 扫描已下载的论文 + 索引状态
    from pathlib import Path
    from src.rag import get_vector_store

    PAPERS_DIR = Path(__file__).parent / "papers"
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))

    # 获取已索引的论文 ID 集合
    store = get_vector_store()
    indexed_ids = set()
    try:
        all_meta = store.collection.get(include=["metadatas"])
        if all_meta["metadatas"]:
            indexed_ids = {m["arxiv_id"] for m in all_meta["metadatas"] if m}
    except Exception:
        pass

    col1, col2 = st.columns(2)
    with col1:
        st.metric("已下载", len(pdfs))
    with col2:
        st.metric("已索引", len(indexed_ids))

    if pdfs:
        if st.button("🔍 一键索引全部", use_container_width=True):
            with st.spinner("正在索引..."):
                store.index_all_downloaded()
            st.rerun()

        st.caption(f"共 {len(pdfs)} 篇论文：")
        for pdf_path in pdfs:
            aid = pdf_path.stem
            is_indexed = aid in indexed_ids
            icon = "📄" if is_indexed else "📄"
            status = "已索引" if is_indexed else "待索引"
            if is_indexed:
                st.caption(f"{icon} `{aid[:20]}` {status}")
            else:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.caption(f"📄 `{aid[:20]}` {status}")
                with col_b:
                    if st.button("索引", key=f"idx_{aid}", use_container_width=True):
                        from src.tools import parse_paper
                        text = parse_paper(aid)
                        if not text.startswith("Error"):
                            store.add_paper(aid, text, {"title": text.split(chr(10))[0][:100]})
                            st.rerun()
    else:
        st.caption("暂无已下载论文")

    # 会话管理
    st.divider()
    st.subheader("💬 会话")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 新会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
    with col2:
        if st.button("🗑️ 清屏", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # 快捷命令
    st.divider()
    st.subheader("⚡ 快捷命令")
    quick_cmds = [
        "search attention mechanism in LLMs",
        "download 1706.03762",
        "summarize 1706.03762",
        "deep read 1706.03762",
        "search_library transformer architecture",
        "library stats",
    ]
    for cmd in quick_cmds:
        if st.button(cmd, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": cmd})
            st.rerun()

# ---- 主区域 ----
st.title("📚 Paper Reading Agent")
st.caption("AI 驱动的学术论文阅读助手 — 搜索、下载、总结、深度分析、RAG 语义检索")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("输入指令，例如：搜索 transformer attention 论文"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent
    with st.chat_message("assistant"):
        try:
            agent = load_agent()
            from langchain_core.messages import HumanMessage

            with st.spinner("🤔 思考中..."):
                result = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                # 只取最后一条 AI 消息
                ai_messages = [m for m in result["messages"] if m.type == "ai"]
                if ai_messages:
                    reply = ai_messages[-1].content
                else:
                    reply = "（Agent 未返回文本回复）"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            import traceback
            st.error(f"❌ 调用失败:\n```\n{traceback.format_exc()}\n```")
            st.session_state.messages.append({"role": "assistant", "content": f"出错了: {e}"})

# ---- 底部状态栏 ----
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"💬 {len(st.session_state.messages)} 条消息")
with col2:
    st.caption(f"🧵 Thread: {st.session_state.thread_id[:12]}...")
with col3:
    try:
        from src.rag import get_vector_store
        st.caption(get_vector_store().get_stats())
    except Exception:
        st.caption("📦 RAG 未初始化")
