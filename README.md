# 📚 Paper Reading Agent

> 基于 **LangGraph + ReAct + RAG** 的多工具论文阅读智能助手  
> 支持论文搜索、下载、总结、深度分析、问答，以及本地向量库语义检索
> 内置精心设计的 **Streamlit 本地研究工作台**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20V4-536DFE.svg)](https://platform.deepseek.com/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 项目概述

Paper Reading Agent 是一个 AI 驱动的学术论文阅读助手。它使用 **ReAct（Reasoning + Acting）范式**，让 LLM 自主决定何时搜索、下载、阅读或分析论文。内置 **RAG 本地向量库**，支持对已下载的论文进行语义检索，实现跨论文的知识问答。

**核心能力：**

- 🔍 **arXiv 论文搜索** — 按关键词检索，返回格式化结果
- 📥 **论文下载** — 一键下载 PDF 到本地
- 📝 **论文总结** — 调用 DeepSeek 生成结构化中文摘要
- 🔬 **深度分析** — 7 维度结构化精读（方法、实验、贡献、局限等）
- 💬 **论文问答** — 基于论文原文的精准问答
- 🧠 **RAG 语义检索** — 本地论文向量化，用自然语言搜相关段落
- 🖥️ **本地研究工作台** — 响应式对话界面、论文库状态、快捷任务与会话管理

### Streamlit 工作台

Web UI 以研究任务为中心，首屏不会加载 Embedding 模型；只有在执行 Agent
任务或点击索引时，才会按需初始化相关资源。

- **清晰的研究入口**：论文发现、经典精读、本地知识检索三个快捷任务
- **对话式工作流**：在一个会话中连续完成搜索、下载、总结、追问与对比
- **本地论文库**：查看已下载 PDF，并一键建立 ChromaDB 向量索引
- **论文标题识别**：侧边栏从 PDF 首页提取标题，并区分已索引与待索引状态
- **浅色 / 深色主题**：在右上角设置菜单中切换，两套配色均针对阅读场景设计
- **安全的配置状态**：只显示 API 是否就绪，不在界面暴露 Key 内容
- **独立会话记忆**：新建会话会创建新的 LangGraph Thread

---

## 🏗️ 架构设计

```
┌──────────────────────────────────────────┐
│     app.py (Streamlit) / main.py (CLI)   │  ← 用户交互层
│       Web 界面 · 终端 Rich 渲染          │
└──────────────────┬───────────────────────┘
                   │ invoke
┌──────────────────▼───────────────────────┐
│           agent.py (LangGraph)           │  ← Agent 编排层
│   StateGraph: agent ⇄ tools 循环         │
│   MemorySaver: 多轮记忆 · 条件路由        │
└────┬──────────────┬──────────────────────┘
     │              │
┌────▼────┐   ┌─────▼──────────────────────┐
│ prompts │   │       tools.py (8 Tools)    │  ← 能力层
│  4 模板  │   │  搜索·下载·总结·精读·问答   │
└─────────┘   │  索引·语义搜索·统计         │
              └──┬───┬──────┬───────────────┘
                 │   │      │
        ┌────────▼───▼──┐   ┌▼──────────────┐
        │  arXiv + PDF  │   │    RAG 模块    │
        │  API · PyMuPDF│   │ ChromaDB +     │
        │  DeepSeek API │   │ sentence-      │
        └───────────────┘   │ transformers   │
                            └────────────────┘
```

**LangGraph 图结构：**

```
START → agent_node ⇄ tools_node
         (LLM)        (工具执行)
           │              │
           └── 条件边 ────┘
           (tool_call? → tools : END)
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）

### 2. 安装

```bash
git clone https://github.com/tiankuolu/paper-agent.git
cd paper-agent
pip install -r requirements.txt
```

### 3. 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 4. 运行

**命令行版本：**
```bash
python main.py
```

**Streamlit Web UI（推荐）：**
```bash
streamlit run app.py
```

浏览器会自动打开 [http://localhost:8501](http://localhost:8501)。如果没有自动
打开，请手动访问该地址。

进入工作台后：

1. 确认侧边栏的 DeepSeek API 状态为“已连接”。
2. 选择一个快捷研究任务，或直接在底部输入问题。
3. 下载论文后，可在侧边栏查看本地 PDF，并点击“索引全部论文”。
4. 使用“新建会话”开始独立研究主题；“清空对话”只清理当前界面。
5. 在右上角 `··· → Settings → Theme` 中选择浅色、深色或跟随系统。

> 首次建立 RAG 索引时会加载 `all-MiniLM-L6-v2` Embedding 模型，
> 所需时间取决于本机环境；后续会复用本地缓存。

### 5. 交互示例

```
📚 Paper Reading Agent
An AI-powered academic paper assistant

You > find papers about reinforcement learning for robotics

Agent: Found 5 papers:
[1] Learning Dexterous Manipulation... (2023)
    ID: 2301.12345 | ...
    ...

You > download 2301.12345

Agent: Downloaded: Learning Dexterous In-Hand Manipulation

You > summarize 2301.12345

Agent: ### 2301.12345
1. Title & Authors: ...
2. Core Problem: ...
...

You > deep read 2301.12345

Agent: ## Deep Read: 2301.12345
1. Paper Overview: ...
2. Problem Definition: ...
...

You > index library

Agent: ✅ 2301.12345: 42 chunks ("Learning Dexterous In-Hand Manipulation")

You > what methods do these papers use for sim-to-real transfer?

Agent: 🔍 Found 3 relevant passages:
### [1] from 2301.12345 (similarity: 89.23%)
We employ domain randomization to bridge the sim-to-real gap...
```

---

## 📁 项目结构

```
paper-agent/
├── app.py                   # Streamlit 本地研究工作台（主入口）
├── main.py                  # CLI 终端入口
├── requirements.txt         # 依赖清单
├── .env                    # API Key 配置（需自行创建）
├── papers/                 # 下载的 PDF 存放目录
├── chroma_db/              # ChromaDB 向量库持久化数据
│
└── src/
    ├── __init__.py         # 包声明
    ├── prompts.py          # Prompt 模板（系统提示、总结、精读、问答）
    ├── tools.py            # 8 个工具函数 + 注册表
    ├── agent.py            # LangGraph Agent 编排（ReAct 循环 + 图构建）
    └── rag.py              # RAG 模块（ChromaDB 向量存储 + 语义检索）
```

---

## 🔧 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **Agent 框架** | LangGraph | StateGraph 构建 agent ⇄ tools 循环 |
| **LLM** | DeepSeek V4 (`deepseek-v4-pro`) | OpenAI 兼容接口，支持 Tool Calling |
| **论文搜索** | `arxiv` (2.x) | arXiv API 官方客户端 |
| **PDF 解析** | PyMuPDF (`fitz`) | 提取论文纯文本 |
| **向量数据库** | ChromaDB | 持久化存储，余弦相似度检索 |
| **Embedding** | `all-MiniLM-L6-v2` | 384 维轻量模型，本地运行 |
| **终端渲染** | Rich | Markdown 彩色输出、状态动画 |
| **配置管理** | python-dotenv | 环境变量注入 |

---

## 🛠️ 可用工具 (8 Tools)

| 工具 | 功能 | 参数 |
|------|------|------|
| `search_papers` | 搜索 arXiv 论文 | `query`, `max_results=5` |
| `download_paper` | 下载论文 PDF | `arxiv_id` |
| `summarize_papers` | 生成中文结构化摘要 | `arxiv_ids`（逗号分隔） |
| `deep_read` | 7 维度深度分析 | `arxiv_id` |
| `chat_with_paper` | 基于论文原文问答 | `arxiv_id`, `question` |
| `index_library` | 索引所有已下载 PDF 到向量库 | — |
| `search_library` | 语义搜索本地论文库 | `query`, `n_results=5` |
| `library_stats` | 查看索引统计 | — |

---

## 🎯 核心知识点

本项目涵盖了 AI Agent 开发的多个关键概念：

- **ReAct 范式** — Reasoning + Acting 循环，LLM 自主决策调用工具
- **LangGraph 图编排** — StateGraph 节点 + 条件边 + MemorySaver 记忆
- **Tool Use / Function Calling** — `@tool` 装饰器 + schema 自动生成
- **Prompt Engineering** — 角色设定、结构化输出、链式提示
- **RAG（检索增强生成）** — 文本分块 → Embedding 向量化 → 语义检索 → 增强回答
- **状态管理** — AgentState 消息历史 + `add_messages` 追加策略
- **单例模式** — 全局复用 Agent 实例和向量库实例，避免重复初始化

---

## 🔜 改进方向

- [ ] **Docker 部署** — 一键容器化运行
- [ ] **多源搜索** — 集成 Semantic Scholar、DBLP、PubMed
- [ ] **引用关系图谱** — NetworkX + PyVis 可视化论文引用网络
- [ ] **论文对比分析** — 同时分析多篇论文的异同
- [ ] **持久化 Memory** — SQLite 替代 MemorySaver，支持重启后恢复对话
- [ ] **多模型适配** — 统一接口支持 Claude / GPT / 本地模型
- [ ] **自动推送** — 每日新论文摘要 + 邮件/企业微信通知

---

## 📄 依赖

```
openai>=1.0.0           # LLM API 调用（OpenAI 兼容）
langgraph>=0.2.0         # Agent 图编排
langchain>=0.3.0         # LLM 框架
langchain-openai>=0.2.0  # LangChain OpenAI 集成
pymupdf>=1.24.0          # PDF 文本提取
arxiv>=2.1.0             # arXiv API 客户端
python-dotenv>=1.0.0     # 环境变量管理
rich>=13.0.0             # 终端美化
chromadb>=0.4.0               # 向量数据库
sentence-transformers>=2.2.0  # Embedding 模型
streamlit>=1.60.0             # Web UI（原生浅色 / 深色双主题）
```

---

## 📄 License

MIT © tiankuolu
