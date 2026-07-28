"""Tools for the Paper Reading Agent.
每个工具函数都是 agent 可以调用的"技能"：
  - 搜索/下载论文（arXiv API）
  - 解析 PDF（PyMuPDF）
  - AI 总结/精读/问答（DeepSeek LLM）
  - 本地向量库索引/语义搜索（ChromaDB + sentence-transformers）

工具函数的返回值是字符串，会直接喂回给 LLM 作为 tool result。"""

import os  # 读取环境变量
from dotenv import load_dotenv  # 从 .env 文件加载配置
load_dotenv()  # 把 .env 中的 KEY=VALUE 注入 os.environ
import arxiv  # arXiv 官方 Python 客户端，用于搜索和获取论文元数据
import fitz  # PyMuPDF 库，用于解析 PDF 提取纯文本
import urllib.request  # 标准库 HTTP 下载，用于下载 PDF 文件
from pathlib import Path  # 面向对象的文件路径操作
from openai import OpenAI  # OpenAI 兼容客户端，同时用于 DeepSeek 和千问 VL
from .prompts import SUMMARIZE_PROMPT, DEEP_READ_PROMPT, CHAT_PROMPT  # 导入提示词模板

# ============================================================
# 全局配置
# ============================================================
PAPERS_DIR = Path(__file__).parent.parent / "papers"  # PDF 存储目录：项目根目录/papers
PAPERS_DIR.mkdir(exist_ok=True)  # 如果目录不存在就创建，已存在则跳过

# DeepSeek 客户端：用于文本总结、深度阅读、论文问答
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取 API Key
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # DeepSeek API 地址（OpenAI 兼容）
)

# ============================================================
# 工具 1：搜索论文
# 调用 arXiv API，按关键词搜索，返回格式化的论文列表
# ============================================================
def search_papers(query: str, max_results: int = 5) -> str:
    """Search arXiv and return formatted results."""
    client = arxiv.Client()  # 创建 arXiv API 客户端
    search = arxiv.Search(  # 构建搜索查询
        query=query,  # 搜索关键词，支持 arXiv 查询语法
        max_results=max_results,  # 最多返回几条结果
        sort_by=arxiv.SortCriterion.Relevance  # 按相关性排序
    )
    results = []  # 收集格式化后的论文信息
    for i, paper in enumerate(client.results(search), 1):  # 遍历搜索结果，i 从 1 开始编号
        aid = paper.entry_id.split("/")[-1].rsplit("v", 1)[0]  # 从 URL 中提取纯 arXiv ID（去掉版本号 v1/v2）
        results.append({
            "index": i,  # 序号，方便用户在对话中引用
            "id": aid,  # arXiv ID，如 1706.03762
            "title": paper.title,  # 论文标题
            "authors": [a.name for a in paper.authors][:5],  # 只取前 5 位作者
            "year": paper.published.year,  # 发表年份
            "summary": paper.summary[:500].replace("\n", " "),  # 摘要前 500 字，去掉换行符
        })

    output = f"Found {len(results)} papers:\n\n"  # 结果头部
    for r in results:
        output += f"[{r['index']}] {r['title']} ({r['year']})\n"  # 编号 + 标题 + 年份
        output += f"    ID: {r['id']} | {', '.join(r['authors'])}\n"  # arXiv ID + 作者列表
        output += f"    {r['summary'][:200]}...\n\n"  # 摘要前 200 字预览
    return output  # 返回格式化字符串，LLM 会直接展示给用户

# ============================================================
# 工具 2：下载论文 PDF
# 通过 arXiv ID 下载 PDF 到本地 papers/ 目录
# ============================================================
def download_paper(arxiv_id: str) -> str:
    """Download PDF from arXiv."""
    aid = arxiv_id.strip().split("v")[0]  # 清理空白，去掉可能的版本号后缀
    pdf_path = PAPERS_DIR / f"{aid}.pdf"  # 目标路径：papers/1706.03762.pdf
    if pdf_path.exists():  # 已下载过，跳过重复下载
        return f"Already downloaded: {pdf_path}"

    client = arxiv.Client()  # 创建 arXiv 客户端
    search = arxiv.Search(id_list=[aid])  # 按 ID 精确查找
    paper = next(client.results(search))  # 取第一条结果（ID 查询只会返回一条）
    urllib.request.urlretrieve(paper.pdf_url, str(pdf_path))  # 用标准库下载 PDF，不需要额外依赖
    return f"Downloaded: {paper.title}"  # 返回下载成功的论文标题

# ============================================================
# 解析 PDF 工具函数（非 agent tool，被其他工具调用）
# 用 PyMuPDF 把 PDF 提取为纯文本，受 max_chars 限制防止 token 爆炸
# ============================================================
def parse_paper(arxiv_id: str, max_chars: int = 30000) -> str:
    """Parse a downloaded PDF into plain text, capped at max_chars."""
    aid = arxiv_id.strip().split("v")[0]  # 清理 arXiv ID
    pdf_path = PAPERS_DIR / f"{aid}.pdf"  # 定位 PDF 文件
    if not pdf_path.exists():  # 还没下载，返回错误
        return f"Error: Paper {aid} not downloaded yet."
    doc = fitz.open(str(pdf_path))  # 用 PyMuPDF 打开 PDF
    text = ""  # 累积提取的文本
    for page in doc:  # 逐页读取
        text += page.get_text()  # 提取当前页的纯文本
        if len(text) > max_chars:  # 超过上限就停止，避免传给 LLM 的文本过长
            break
    doc.close()  # 关闭 PDF 文件句柄，释放资源
    return text[:max_chars]  # 截断到 max_chars

def _get_text(aid): return parse_paper(aid)  # 简写，内部调用用

# ============================================================
# 工具 3：总结论文
# 支持逗号分隔的多个 arXiv ID，逐个解析 PDF → 调用 DeepSeek 生成中文总结
# ============================================================
def summarize_papers(arxiv_ids: str) -> str:
    """Summarize one or more papers (comma-separated arXiv IDs) in Chinese."""
    output = ""  # 累积所有论文的总结
    for aid in arxiv_ids.split(","):  # 按逗号拆分多个 ID
        aid = aid.strip().split("v")[0]  # 清理每个 ID
        text = _get_text(aid)  # 解析 PDF 为文本
        if text.startswith("Error"):  # PDF 解析失败
            output += f"### {aid}\n{text}\n---\n"
            continue
        r = client.chat.completions.create(  # 调用 DeepSeek 做总结
            model="deepseek-v4-pro",  # 使用的模型
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(paper_text=text[:8000])}],  # 前 8000 字 + 总结模板
            temperature=0.3,  # 低温度，输出更确定、更一致
            max_tokens=1500  # 限制输出长度
        )
        output += f"### {aid}\n{r.choices[0].message.content}\n---\n"  # 拼接结果
    return output

# ============================================================
# 工具 4：深度阅读
# 对单篇论文做 7 维度结构化深度分析
# ============================================================
def deep_read(arxiv_id: str) -> str:
    """Deep structured analysis (7 dimensions) of a single paper."""
    aid = arxiv_id.strip().split("v")[0]  # 清理 ID
    text = _get_text(aid)  # 解析 PDF
    if text.startswith("Error"): return text  # 解析失败直接返回错误
    r = client.chat.completions.create(  # 调用 DeepSeek 做深度分析
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": DEEP_READ_PROMPT.format(paper_text=text[:15000])}],  # 前 15000 字 + 深度分析模板
        temperature=0.3,
        max_tokens=3000  # 深度分析需要更多 token
    )
    return f"## Deep Read: {aid}\n\n{r.choices[0].message.content}"

# ============================================================
# 工具 5：论文问答
# 基于论文全文回答用户的具体问题，要求模型只基于原文、不编造
# ============================================================
def chat_with_paper(arxiv_id: str, question: str) -> str:
    """Answer a question based solely on the paper content."""
    aid = arxiv_id.strip().split("v")[0]  # 清理 ID
    text = _get_text(aid)  # 解析 PDF
    if text.startswith("Error"): return text  # 解析失败直接返回
    r = client.chat.completions.create(  # 调用 DeepSeek 做问答
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": CHAT_PROMPT.format(
            paper_text=text[:15000],  # 论文全文（截断）
            question=question  # 用户问题
        )}],
        temperature=0.3,
        max_tokens=1500
    )
    return f"Q: {question}\n\n{r.choices[0].message.content}"

# ============================================================
# 工具 6：索引本地论文库
# 扫描 papers/ 目录下所有 PDF → 提取文本 → 切片 → 向量化 → 存入 ChromaDB
# ============================================================
def index_library() -> str:
    """Index all downloaded PDFs in papers/ into the local vector library for semantic search."""
    from .rag import get_vector_store  # 延迟导入，避免循环依赖
    return get_vector_store().index_all_downloaded()  # 委托给 PaperVectorStore

# ============================================================
# 工具 7：语义搜索本地论文库
# 用自然语言查询，在已索引的论文中找最相关的段落
# ============================================================
def search_library(query: str, n_results: int = 5) -> str:
    """Semantic search across locally indexed papers. Find relevant passages."""
    from .rag import get_vector_store  # 延迟导入
    results = get_vector_store().search(query, n_results)  # 向量相似度搜索
    if not results:  # 没有匹配结果
        return "No results found. Try indexing papers first with 'index_library'."

    output = f"🔍 Found {len(results)} relevant passages:\n\n"  # 结果头部
    for i, r in enumerate(results, 1):
        # 余弦距离转相似度百分比（ChromaDB 用余弦距离，1-距离 = 相似度）
        output += f"### [{i}] from {r['arxiv_id']}  (similarity: {1 - (r['distance'] or 0):.2%})\n"
        output += f"{r['content'][:400]}\n\n"  # 只展示前 400 字
    return output

# ============================================================
# 工具 8：查看索引统计
# 显示当前 ChromaDB 中有多少论文、多少文本块
# ============================================================
def library_stats() -> str:
    """Show how many papers and chunks are indexed in the local library."""
    from .rag import get_vector_store  # 延迟导入
    return get_vector_store().get_stats()  # 返回统计字符串

# ============================================================
# 工具注册表：名称 → 函数映射
# 方便按名称查找工具函数
# ============================================================
TOOL_REGISTRY = {
    "search_papers": search_papers,
    "download_paper": download_paper,
    "parse_paper": parse_paper,  # 内部工具，不在 agent tool 列表中但可被其他工具调用
    "summarize_papers": summarize_papers,
    "deep_read": deep_read,
    "chat_with_paper": chat_with_paper,
    "index_library": index_library,
    "search_library": search_library,
    "library_stats": library_stats,
}
