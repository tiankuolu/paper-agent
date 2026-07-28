"""Polished local Streamlit UI for Paper Reading Agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import traceback
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
PAPERS_DIR = APP_ROOT / "papers"

st.set_page_config(
    page_title="Paper Reading Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_STYLES = """
<style>
    :root {
        --paper-bg: var(--background-color);
        --paper-surface: color-mix(in srgb, var(--secondary-background-color) 82%, transparent);
        --paper-ink: var(--text-color);
        --paper-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
        --paper-line: var(--border-color);
        --paper-accent: var(--primary-color);
        --paper-accent-2: var(--green-color);
    }

    html,
    body,
    .stApp {
        background: var(--background-color) !important;
        color: var(--text-color) !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 82% 2%,
                color-mix(in srgb, var(--primary-color) 11%, transparent),
                transparent 30rem
            ),
            radial-gradient(
                circle at 12% 92%,
                color-mix(in srgb, var(--green-color) 8%, transparent),
                transparent 28rem
            ),
            var(--paper-bg);
        color: var(--text-color);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }

    [data-testid="stSidebar"] {
        background:
            radial-gradient(
                circle at 12% 0%,
                color-mix(in srgb, var(--primary-color) 11%, transparent),
                transparent 18rem
            ),
            var(--secondary-background-color);
        border-right: 1px solid var(--border-color);
        color: var(--text-color);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.65rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: var(--text-color);
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--paper-muted);
    }

    .paper-brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1.35rem;
    }

    .paper-brand-mark {
        display: grid;
        width: 2.55rem;
        height: 2.55rem;
        place-items: center;
        border: 1px solid color-mix(in srgb, var(--primary-color) 42%, var(--border-color));
        border-radius: 0.85rem;
        background: linear-gradient(145deg, var(--primary-color), var(--green-color));
        box-shadow: 0 10px 28px color-mix(in srgb, var(--primary-color) 22%, transparent);
        color: white;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .paper-brand-name {
        color: var(--text-color);
        font-size: 1.04rem;
        font-weight: 720;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }

    .paper-brand-meta {
        margin-top: 0.18rem;
        color: var(--paper-muted);
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.1rem 0 1rem;
        padding: 0.72rem 0.8rem;
        border: 1px solid var(--border-color);
        border-radius: 0.8rem;
        background: color-mix(in srgb, var(--background-color) 56%, transparent);
    }

    .status-label {
        color: var(--paper-muted);
        font-size: 0.77rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: var(--text-color);
        font-size: 0.74rem;
        font-weight: 650;
    }

    .status-dot {
        width: 0.48rem;
        height: 0.48rem;
        border-radius: 999px;
        background: var(--orange-color);
        box-shadow: 0 0 0 0.22rem color-mix(in srgb, var(--orange-color) 14%, transparent);
    }

    .status-dot.ready {
        background: var(--green-color);
        box-shadow: 0 0 0 0.22rem color-mix(in srgb, var(--green-color) 14%, transparent);
    }

    .sidebar-section-label {
        margin: 1.2rem 0 0.55rem;
        color: var(--paper-muted);
        font-size: 0.68rem;
        font-weight: 720;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        padding: 0.72rem 0.78rem;
        border: 1px solid var(--border-color);
        border-radius: 0.8rem;
        background: color-mix(in srgb, var(--background-color) 52%, transparent);
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.38rem;
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
    }

    [data-testid="stButton"] > button {
        min-height: 2.55rem;
        width: 100%;
        border: 1px solid var(--paper-line);
        border-radius: 0.82rem;
        background: color-mix(in srgb, var(--background-color) 86%, transparent);
        color: var(--paper-ink);
        font-weight: 650;
        box-shadow: 0 5px 18px color-mix(in srgb, var(--text-color) 6%, transparent);
        transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }

    [data-testid="stButton"] > button:hover {
        border-color: color-mix(in srgb, var(--primary-color) 52%, var(--border-color));
        color: var(--primary-color);
        box-shadow: 0 9px 22px color-mix(in srgb, var(--primary-color) 14%, transparent);
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        border-color: var(--border-color);
        background: color-mix(in srgb, var(--background-color) 52%, transparent);
        color: var(--text-color);
        box-shadow: none;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        border-color: color-mix(in srgb, var(--primary-color) 58%, var(--border-color));
        background: color-mix(in srgb, var(--primary-color) 12%, var(--background-color));
        color: var(--primary-color);
    }

    .paper-hero {
        position: relative;
        overflow: hidden;
        margin-bottom: 1.35rem;
        padding: clamp(1.7rem, 4vw, 3rem);
        border: 1px solid var(--border-color);
        border-radius: 1.45rem;
        background:
            radial-gradient(
                circle at 86% 18%,
                color-mix(in srgb, var(--green-color) 15%, transparent),
                transparent 18rem
            ),
            radial-gradient(
                circle at 70% 100%,
                color-mix(in srgb, var(--primary-color) 18%, transparent),
                transparent 24rem
            ),
            linear-gradient(
                135deg,
                color-mix(in srgb, var(--primary-color) 8%, var(--background-color)) 0%,
                var(--secondary-background-color) 100%
            );
        box-shadow: 0 24px 70px color-mix(in srgb, var(--text-color) 13%, transparent);
    }

    .paper-hero::after {
        position: absolute;
        top: -5rem;
        right: -4rem;
        width: 15rem;
        height: 15rem;
        border: 1px solid color-mix(in srgb, var(--primary-color) 18%, transparent);
        border-radius: 50%;
        content: "";
    }

    .paper-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.48rem;
        margin-bottom: 1rem;
        color: var(--paper-muted);
        font-size: 0.71rem;
        font-weight: 730;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .paper-eyebrow::before {
        width: 1.6rem;
        height: 1px;
        background: var(--green-color);
        content: "";
    }

    .paper-hero h1 {
        max-width: 760px;
        margin: 0;
        color: var(--text-color);
        font-size: clamp(2.2rem, 5vw, 4rem);
        font-weight: 730;
        letter-spacing: -0.058em;
        line-height: 1.02;
    }

    .paper-hero p {
        max-width: 670px;
        margin: 1rem 0 0;
        color: var(--paper-muted);
        font-size: 1rem;
        line-height: 1.7;
    }

    .hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.35rem;
    }

    .hero-chip {
        padding: 0.42rem 0.7rem;
        border: 1px solid var(--border-color);
        border-radius: 999px;
        background: color-mix(in srgb, var(--background-color) 58%, transparent);
        color: var(--text-color);
        font-size: 0.76rem;
    }

    .section-heading {
        margin: 1.4rem 0 0.7rem;
        color: var(--paper-ink);
        font-size: 1rem;
        font-weight: 730;
        letter-spacing: -0.02em;
    }

    .starter-card {
        min-height: 9.5rem;
        padding: 1.15rem;
        border: 1px solid var(--paper-line);
        border-radius: 1rem;
        background: var(--paper-surface);
        box-shadow: 0 12px 32px color-mix(in srgb, var(--text-color) 7%, transparent);
        backdrop-filter: blur(12px);
    }

    .starter-number {
        color: var(--paper-accent);
        font-size: 0.68rem;
        font-weight: 760;
        letter-spacing: 0.12em;
    }

    .starter-card h3 {
        margin: 0.75rem 0 0.45rem;
        color: var(--paper-ink);
        font-size: 1rem;
        letter-spacing: -0.025em;
    }

    .starter-card p {
        margin: 0;
        color: var(--paper-muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 0.7rem;
        padding: 1rem 1.05rem;
        border: 1px solid var(--paper-line);
        border-radius: 1rem;
        background: color-mix(in srgb, var(--background-color) 82%, transparent);
        box-shadow: 0 8px 24px color-mix(in srgb, var(--text-color) 6%, transparent);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: color-mix(in srgb, var(--primary-color) 8%, var(--background-color));
    }

    [data-testid="stChatInput"] {
        border-color: color-mix(in srgb, var(--primary-color) 28%, var(--border-color));
        border-radius: 1rem;
        background: color-mix(in srgb, var(--background-color) 94%, transparent);
        box-shadow: 0 12px 34px color-mix(in srgb, var(--text-color) 11%, transparent);
    }

    [data-testid="stChatInput"] textarea {
        background: transparent;
        color: var(--text-color);
    }

    [data-testid="stBottom"] {
        background: linear-gradient(
            180deg,
            transparent 0%,
            color-mix(in srgb, var(--background-color) 94%, transparent) 26%,
            var(--background-color) 100%
        );
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--paper-line);
        border-radius: 0.85rem;
        background: color-mix(in srgb, var(--background-color) 62%, transparent);
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border-color: var(--border-color);
        background: color-mix(in srgb, var(--background-color) 45%, transparent);
    }

    .paper-footer {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 0.75rem;
        margin-top: 1.25rem;
        padding: 0.8rem 0.1rem 0.1rem;
        border-top: 1px solid var(--paper-line);
        color: var(--paper-muted);
        font-size: 0.73rem;
    }

    .paper-footer strong {
        color: var(--paper-ink);
        font-weight: 650;
    }

    @media (max-width: 800px) {
        .main .block-container {
            padding-top: 1rem;
        }

        .paper-hero {
            border-radius: 1.1rem;
        }

        .paper-hero h1 {
            font-size: 2.35rem;
        }

        .starter-card {
            min-height: auto;
        }
    }
</style>
"""

THEME_TOKENS = {
    "light": {
        "primary": "#5A54D6",
        "background": "#FBFBF8",
        "secondary": "#F1F2F5",
        "text": "#1C2430",
        "border": "#D8DCE4",
        "green": "#168576",
        "orange": "#B35B12",
    },
    "dark": {
        "primary": "#8179EB",
        "background": "#0F1218",
        "secondary": "#191D26",
        "text": "#E7EAF0",
        "border": "#2B313D",
        "green": "#56BFB1",
        "orange": "#E0A15A",
    },
}
active_theme_type = st.context.theme.type or "light"
if active_theme_type not in THEME_TOKENS:
    active_theme_type = "light"
active_tokens = THEME_TOKENS[active_theme_type]
THEME_VARIABLES = f"""
<style>
    :root {{
        --primary-color: {active_tokens["primary"]};
        --background-color: {active_tokens["background"]};
        --secondary-background-color: {active_tokens["secondary"]};
        --text-color: {active_tokens["text"]};
        --border-color: {active_tokens["border"]};
        --green-color: {active_tokens["green"]};
        --orange-color: {active_tokens["orange"]};
    }}
</style>
"""

st.markdown(APP_STYLES, unsafe_allow_html=True)
st.markdown(THEME_VARIABLES, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize all per-browser-tab state in one place."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))


def new_session() -> None:
    """Start a new LangGraph memory thread and clear visible chat."""
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def clear_chat() -> None:
    """Clear visible messages while keeping the current agent thread."""
    st.session_state.messages = []


def get_api_key() -> str:
    """Read the DeepSeek key without ever rendering its value."""
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    try:
        key = str(st.secrets.get("DEEPSEEK_API_KEY", "")).strip()
    except Exception:
        key = ""

    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
    return key


@st.cache_resource(show_spinner=False)
def load_agent():
    """Load the compiled LangGraph agent only after the first question."""
    from src.agent import get_agent

    return get_agent()


@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Load the embedding model and vector store only when indexing is requested."""
    from src.rag import get_vector_store

    return get_vector_store()


@st.cache_data(show_spinner=False, max_entries=256)
def extract_pdf_title(pdf_path: str, modified_ns: int) -> str:
    """Extract a likely paper title from the largest text on page one."""
    del modified_ns  # Part of the cache key so replacing a PDF invalidates its title.

    path = Path(pdf_path)
    fallback = path.stem
    try:
        import fitz

        with fitz.open(pdf_path) as document:
            if document.page_count == 0:
                return fallback

            page = document[0]
            page_height = float(page.rect.height)
            candidates: list[dict[str, object]] = []
            page_dict = page.get_text("dict", sort=True)

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        raw_text = str(span.get("text", ""))
                        text = re.sub(r"\s+", " ", raw_text).strip()
                        if not text:
                            continue

                        lower_text = text.lower()
                        top = float(span.get("bbox", (0, 0, 0, 0))[1])
                        if top > page_height * 0.46:
                            continue
                        if any(
                            marker in lower_text
                            for marker in ("arxiv:", "preprint", "submitted to", "proceedings of")
                        ):
                            continue

                        candidates.append(
                            {
                                "text": text,
                                "size": float(span.get("size", 0)),
                                "x": float(span.get("bbox", (0, 0, 0, 0))[0]),
                                "x2": float(span.get("bbox", (0, 0, 0, 0))[2]),
                                "y": top,
                                "leading_space": raw_text[:1].isspace(),
                            }
                        )

            if candidates:
                word_candidates = [
                    item for item in candidates if len(str(item["text"]).strip()) >= 3
                ]
                largest_size = max(
                    float(item["size"]) for item in (word_candidates or candidates)
                )
                title_spans = [
                    item
                    for item in candidates
                    if float(item["size"]) >= max(12.0, largest_size - 1.25)
                    or (
                        len(str(item["text"]).strip()) <= 2
                        and float(item["size"]) >= largest_size
                    )
                ]
                title_spans.sort(key=lambda item: (float(item["y"]), float(item["x"])))

                lines: list[list[dict[str, object]]] = []
                for item in title_spans:
                    if not lines:
                        lines.append([item])
                        continue

                    line_y = sum(float(part["y"]) for part in lines[-1]) / len(lines[-1])
                    line_tolerance = max(3.0, float(item["size"]) * 0.34)
                    if abs(float(item["y"]) - line_y) <= line_tolerance:
                        lines[-1].append(item)
                    else:
                        lines.append([item])

                title_lines: list[str] = []
                for line in lines:
                    line.sort(key=lambda item: float(item["x"]))
                    assembled = ""
                    previous_right: float | None = None
                    for item in line:
                        text = str(item["text"]).strip()
                        if not text:
                            continue
                        if previous_right is not None and assembled:
                            gap = float(item["x"]) - previous_right
                            join_threshold = 1.5
                            if bool(item["leading_space"]) or gap > join_threshold:
                                assembled += " "
                        assembled += text
                        previous_right = float(item["x2"])
                    if assembled:
                        title_lines.append(assembled)

                title = re.sub(r"\s+", " ", " ".join(title_lines)).strip(" -–—")
                if 8 <= len(title) <= 240:
                    return title

            for line in page.get_text("text", sort=True).splitlines():
                title = re.sub(r"\s+", " ", line).strip()
                if len(title) >= 8 and "arxiv:" not in title.lower():
                    return title[:240]
    except Exception:
        pass

    return fallback


@st.cache_data(ttl=10, show_spinner=False)
def scan_downloaded_papers(directory: str) -> list[dict[str, object]]:
    """Return local PDF metadata and titles without initializing the vector model."""
    paper_dir = Path(directory)
    if not paper_dir.exists():
        return []

    papers: list[dict[str, object]] = []
    for path in sorted(paper_dir.glob("*.pdf"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        papers.append(
            {
                "arxiv_id": path.stem,
                "title": extract_pdf_title(str(path), stat.st_mtime_ns),
                "size_mb": stat.st_size / 1024 / 1024,
            }
        )
    return papers


@st.cache_data(ttl=10, show_spinner=False)
def scan_indexed_paper_ids(directory: str) -> list[str]:
    """Read Chroma metadata without loading the sentence-transformer model."""
    chroma_path = Path(directory)
    if not chroma_path.exists():
        return []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(name="papers")
        payload = collection.get(include=["metadatas"])
        return sorted(
            {
                str(metadata["arxiv_id"])
                for metadata in payload.get("metadatas") or []
                if metadata and metadata.get("arxiv_id")
            }
        )
    except Exception:
        return []


def normalize_reply(content: object) -> str:
    """Convert LangChain message content into displayable Markdown."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                text_parts.append(str(item["text"]))
        if text_parts:
            return "\n\n".join(text_parts)
    return str(content)


def invoke_agent(prompt: str) -> str:
    """Run one user turn and return the latest AI message."""
    from langchain_core.messages import HumanMessage

    result = load_agent().invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", "") == "ai":
            return normalize_reply(message.content)
    return "Agent 已完成处理，但没有返回可显示的文本。"


init_session_state()
api_ready = bool(get_api_key())
downloaded_papers = scan_downloaded_papers(str(PAPERS_DIR))
indexed_ids = set(scan_indexed_paper_ids(str(APP_ROOT / "chroma_db")))
current_theme = "深色" if active_theme_type == "dark" else "浅色"


with st.sidebar:
    st.markdown(
        """
        <div class="paper-brand">
            <div class="paper-brand-mark">PA</div>
            <div>
                <div class="paper-brand-name">Paper agent</div>
                <div class="paper-brand-meta">Local research workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    index_notice = st.session_state.pop("index_notice", None)
    if index_notice:
        st.toast(index_notice)

    st.markdown('<div class="sidebar-section-label">外观</div>', unsafe_allow_html=True)
    st.caption(f"当前为{current_theme}模式 · 可在右上角 ··· → Settings → Theme 切换")

    connection_state = "已连接" if api_ready else "等待配置"
    dot_class = "status-dot ready" if api_ready else "status-dot"
    st.markdown(
        f"""
        <div class="status-row">
            <span class="status-label">DeepSeek API</span>
            <span class="status-pill"><span class="{dot_class}"></span>{connection_state}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_ready:
        with st.expander("配置 API Key"):
            st.caption("在项目根目录创建 `.env` 文件，然后刷新页面。")
            st.code(
                "DEEPSEEK_API_KEY=sk-your-key\n"
                "DEEPSEEK_BASE_URL=https://api.deepseek.com",
                language="bash",
            )

    st.markdown('<div class="sidebar-section-label">本地论文库</div>', unsafe_allow_html=True)
    metric_left, metric_right = st.columns(2)
    metric_left.metric("已下载", len(downloaded_papers))
    indexed_downloads = sum(
        1 for paper in downloaded_papers if str(paper["arxiv_id"]) in indexed_ids
    )
    metric_right.metric("已索引", indexed_downloads)

    if downloaded_papers:
        if st.button("索引全部论文", key="index_all"):
            try:
                with st.status("正在加载本地向量库并建立索引…", expanded=True) as status:
                    report = load_vector_store().index_all_downloaded()
                    st.markdown(report)
                    status.update(label="论文库索引完成", state="complete", expanded=False)
                scan_indexed_paper_ids.clear()
                st.session_state.index_notice = "本地论文库已更新"
                st.rerun()
            except Exception as exc:
                st.error(f"索引失败：{exc}")

        with st.expander(f"查看本地论文（{len(downloaded_papers)}）"):
            paper_rows = sorted(
                downloaded_papers,
                key=lambda paper: (
                    str(paper["arxiv_id"]) not in indexed_ids,
                    str(paper["title"]).casefold(),
                ),
            )
            for paper in paper_rows:
                arxiv_id = str(paper["arxiv_id"])
                title = str(paper["title"])
                status_badge = (
                    ":green-badge[已索引]" if arxiv_id in indexed_ids else ":orange-badge[待索引]"
                )
                st.markdown(f"{status_badge}  \n**{title}**")
                st.caption(f"arXiv {arxiv_id} · PDF · {paper['size_mb']:.1f} MB")
    else:
        st.caption("还没有下载论文。先在对话中搜索并下载一篇。")

    st.markdown('<div class="sidebar-section-label">当前会话</div>', unsafe_allow_html=True)
    st.caption(f"Thread · {st.session_state.thread_id[:8]}")
    session_left, session_right = st.columns(2)
    session_left.button("新建会话", key="new_session", on_click=new_session)
    session_right.button("清空对话", key="clear_chat", on_click=clear_chat)

    st.markdown('<div class="sidebar-section-label">隐私</div>', unsafe_allow_html=True)
    st.caption("PDF、向量索引与会话界面均保留在本机；模型请求发送至你配置的 DeepSeek 接口。")


st.markdown(
    """
    <section class="paper-hero">
        <div class="paper-eyebrow">Research copilot · Local first</div>
        <h1>把论文读懂，<br>而不是读完。</h1>
        <p>
            从 arXiv 搜索到结构化精读，再到跨论文语义检索。
            让 Agent 帮你定位证据、拆解方法，并把本地论文变成可追问的知识库。
        </p>
        <div class="hero-chips">
            <span class="hero-chip">arXiv 搜索</span>
            <span class="hero-chip">PDF 精读</span>
            <span class="hero-chip">多轮问答</span>
            <span class="hero-chip">本地 RAG</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


quick_prompt: str | None = None

if not st.session_state.messages:
    st.markdown('<div class="section-heading">从一个研究任务开始</div>', unsafe_allow_html=True)
    action_columns = st.columns(3)
    starter_cards = [
        (
            "01 · Discover",
            "追踪一个研究方向",
            "从 arXiv 找到高相关论文，并整理标题、作者与研究问题。",
            "搜索 AI Agent 的近期论文",
            "搜索 5 篇关于 AI Agent 规划与工具使用的高相关论文",
        ),
        (
            "02 · Understand",
            "拆解经典方法",
            "下载论文后，按问题、方法、实验、贡献与局限进行结构化精读。",
            "精读 Transformer",
            "下载 1706.03762，并从方法、实验、贡献和局限四方面深度解读",
        ),
        (
            "03 · Synthesize",
            "检索本地知识",
            "在已索引论文中定位相关段落，汇总证据并回答跨论文问题。",
            "检索本地论文库",
            "检索本地论文库中与检索增强生成评估方法相关的内容",
        ),
    ]

    for column, (number, title, description, button_label, prompt_text) in zip(
        action_columns, starter_cards
    ):
        with column:
            st.markdown(
                f"""
                <div class="starter-card">
                    <div class="starter-number">{number}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(button_label, key=f"starter_{number}"):
                quick_prompt = prompt_text


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


typed_prompt = st.chat_input("输入研究任务，例如：比较这两篇论文的核心方法")
prompt = typed_prompt or quick_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_ready:
            reply = (
                "还没有检测到 DeepSeek API Key。请在项目根目录的 `.env` 文件中配置 "
                "`DEEPSEEK_API_KEY`，保存后刷新页面即可开始。"
            )
            st.warning(reply)
        else:
            try:
                with st.status("正在理解任务并规划工具调用…", expanded=False) as status:
                    reply = invoke_agent(prompt)
                    status.update(label="研究任务已完成", state="complete", expanded=False)
                st.markdown(reply)
                scan_downloaded_papers.clear()
                scan_indexed_paper_ids.clear()
            except Exception as exc:
                reply = f"请求未完成：{exc}"
                st.error(reply)
                with st.expander("查看诊断信息"):
                    st.code(traceback.format_exc(), language="text")

    st.session_state.messages.append({"role": "assistant", "content": reply})


st.markdown(
    f"""
    <div class="paper-footer">
        <span><strong>{len(st.session_state.messages)}</strong> 条消息</span>
        <span>Thread <strong>{st.session_state.thread_id[:12]}</strong></span>
        <span>LangGraph · ReAct · ChromaDB</span>
    </div>
    """,
    unsafe_allow_html=True,
)
