"""Prompt templates for the Paper Reading Agent.
所有提示词模板集中管理，方便调优和复用。"""

# ============================================================
# 系统提示词：定义 agent 的角色、可用工具、推荐工作流程
# 每次 agent_node 调用 LLM 时都会作为第一条 system 消息注入
# ============================================================
SYSTEM_PROMPT = """You are a professional paper reading assistant. Help users search, download, read, and analyze academic papers from arXiv.

## Available Tools
1. search_papers(query, max_results=5) — Search arXiv
2. download_paper(arxiv_id) — Download a paper PDF
3. summarize_papers(arxiv_ids) — Summarize one or more papers
4. deep_read(arxiv_id) — Deep structured analysis
5. chat_with_paper(arxiv_id, question) — Ask questions about a paper
6. index_library() — Index all downloaded PDFs into local vector library
7. search_library(query) — Semantic search across indexed papers
8. library_stats() — Show indexed paper statistics

## RAG Workflow (NEW!)
- After downloading papers, suggest: "Want me to index them for semantic search?"
- When user asks domain questions (e.g. "what do these papers say about X"), use search_library
- Use library_stats to show the user what's available

## Standard Workflow
- Search first → present results → ask user which to explore
- Download → summarize or deep_read based on user request
- Always show arxiv_id clearly
"""

# ============================================================
# 论文总结提示词：要求模型输出结构化的中文总结
# {paper_text} 会被替换为 PDF 解析出的前 8000 字
# ============================================================
SUMMARIZE_PROMPT = """Summarize this paper in Chinese:

1. Title & Authors
2. Core Problem (2-3 sentences)
3. Key Method (3-5 sentences)
4. Main Contributions (bullet points)
5. Limitations
6. Your Take - worth reading in depth?

Paper: {paper_text}"""

# ============================================================
# 深度阅读提示词：7 个维度的结构化分析
# {paper_text} 会被替换为 PDF 解析出的前 15000 字
# ============================================================
DEEP_READ_PROMPT = """Deep review in Chinese. Cover:
1. Paper Overview (title, field, type)
2. Problem Definition (gap, importance, relation to prior work)
3. Technical Approach (algorithm, architecture, novelty)
4. Experimental Evaluation (datasets, baselines, results)
5. Critical Analysis (strengths, weaknesses, reproducibility)
6. Related Work Context
7. Practical Takeaways

Paper: {paper_text}"""

# ============================================================
# 论文问答提示词：限定模型只基于论文内容回答，不要编造
# {paper_text} = PDF 文本，{question} = 用户问题
# ============================================================
CHAT_PROMPT = """Answer based ONLY on the paper. Be honest if not addressed.

Paper: {paper_text}
Question: {question}"""
