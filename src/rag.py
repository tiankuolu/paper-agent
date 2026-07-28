"""RAG module — local paper vector library using ChromaDB + sentence-transformers.
实现论文的本地向量索引和语义搜索：
  1. 把 PDF 文本切成重叠的段落块（chunk）
  2. 用 sentence-transformers 把每个 chunk 编码为向量
  3. 存入 ChromaDB（持久化到磁盘）
  4. 查询时，把用户问题也向量化，用余弦相似度找最相关的 chunk"""

from pathlib import Path  # 文件路径操作
from typing import List, Dict, Optional  # 类型注解

import chromadb  # 向量数据库，支持持久化存储和相似度检索
from chromadb.utils import embedding_functions  # ChromaDB 内置的 embedding 函数包装器
import fitz  # PyMuPDF，解析 PDF 提取文本

# ============================================================
# 路径配置
# ============================================================
PAPERS_DIR = Path(__file__).parent.parent / "papers"  # PDF 存放目录
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"  # ChromaDB 持久化数据目录


class PaperVectorStore:
    """论文向量存储：管理 PDF → 文本 → 向量 → 检索的完整管线。"""

    def __init__(self):
        """初始化 ChromaDB 客户端和 embedding 模型。"""
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # 持久化客户端，数据存在 chroma_db/ 目录
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # 轻量级 sentence-transformers 模型，384 维向量，适合本地运行
        )
        self.collection = self.client.get_or_create_collection(  # 获取或创建名为 "papers" 的集合
            name="papers",
            embedding_function=self.embedding_fn,  # 指定 embedding 函数，自动在添加/查询时向量化
            metadata={"hnsw:space": "cosine"}  # 用余弦距离做相似度度量
        )

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """把长文本切成有重叠的段落块，尽量在段落边界处断开。

        参数:
            text: 原始文本
            chunk_size: 每个块的目标最大字符数
            overlap: 相邻块之间的重叠字符数

        返回:
            文本块列表"""
        paragraphs = text.split("\n\n")  # 按空行分段，保持段落完整性
        chunks = []  # 收集所有切好的块
        current = ""  # 当前正在构建的块
        for para in paragraphs:
            para = para.strip()  # 去掉首尾空白
            if not para:  # 跳过空段落
                continue
            if len(current) + len(para) > chunk_size and current:  # 当前块已满
                chunks.append(current.strip())  # 保存当前块
                # 从当前块末尾取 overlap 长度的词作为下一块的前缀，实现重叠
                words = current.split()
                overlap_text = " ".join(words[-max(1, overlap // 10):])  # 取末尾约 overlap 个字符的词
                current = overlap_text + "\n\n" + para  # 新块 = 重叠前缀 + 新段落
            else:
                # 块未满，直接追加
                current = (current + "\n\n" + para).strip() if current else para
        if current.strip():  # 最后一个块（可能不满 chunk_size）
            chunks.append(current.strip())
        return chunks if chunks else [text[:chunk_size]]  # 如果切不出块（无空行），直接用前 chunk_size 字

    def add_paper(self, arxiv_id: str, text: str, metadata: Optional[Dict] = None) -> int:
        """把一篇论文的文本向量化并添加到 ChromaDB。

        参数:
            arxiv_id: 论文 arXiv ID，用于去重和追溯
            text: 论文全文
            metadata: 可选的元数据字典（如标题）

        返回:
            添加的 chunk 数量（0 表示已索引过，跳过）"""
        existing = self.collection.get(where={"arxiv_id": arxiv_id})  # 检查是否已索引过
        if existing and existing["ids"]:  # 已经存在，去重跳过
            return 0

        chunks = self._split_text(text)  # 切分成重叠段落块
        meta = metadata or {}  # 元数据默认为空字典
        ids = [f"{arxiv_id}_chunk_{i}" for i in range(len(chunks))]  # 每块一个唯一 ID，格式：1706.03762_chunk_0
        metadatas = [{**meta, "arxiv_id": arxiv_id, "chunk_index": i} for i in range(len(chunks))]  # 每块的元数据，含论文 ID 和序号

        self.collection.add(  # 批量添加到 ChromaDB（会自动调用 embedding_fn 向量化）
            documents=chunks,  # 原始文本
            ids=ids,  # 唯一 ID
            metadatas=metadatas  # 元数据
        )
        return len(chunks)  # 返回添加了多少块

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """语义搜索：用自然语言查询，返回最相关的论文段落。

        参数:
            query: 查询文本（自然语言）
            n_results: 返回几条最相关的结果

        返回:
            结果列表，每项包含 content（文本）、arxiv_id（来源）、distance（距离）"""
        results = self.collection.query(  # ChromaDB 查询：自动向量化 query，计算余弦距离
            query_texts=[query],
            n_results=n_results
        )

        output = []  # 格式化输出
        if results["documents"] and results["documents"][0]:  # 有结果
            for i, doc in enumerate(results["documents"][0]):  # 遍历返回的文档
                meta = results["metadatas"][0][i] if results["metadatas"] else {}  # 取对应元数据
                dist = results["distances"][0][i] if results.get("distances") else None  # 取余弦距离
                output.append({
                    "content": doc[:500],  # 截断到 500 字展示
                    "arxiv_id": meta.get("arxiv_id", "unknown"),  # 来源论文 ID
                    "distance": round(dist, 4) if dist else None  # 余弦距离保留 4 位小数
                })
        return output

    def index_all_downloaded(self) -> str:
        """扫描 papers/ 目录下所有 PDF，逐个解析并索引到向量库。

        返回:
            索引进度报告，显示每篇论文的状态（成功/已存在/失败）"""
        pdfs = list(PAPERS_DIR.glob("*.pdf"))  # 获取所有 PDF 文件
        if not pdfs:  # 目录为空
            return "No PDFs found in papers/ directory. Download some papers first!"

        results = []  # 每篇论文的索引进度
        for pdf_path in pdfs:
            arxiv_id = pdf_path.stem  # 文件名去掉 .pdf 后缀 = arXiv ID
            try:
                doc = fitz.open(str(pdf_path))  # 打开 PDF
                text = ""  # 累积全文
                for page in doc:  # 逐页提取文本
                    text += page.get_text()
                doc.close()  # 关闭文件

                title = text.split("\n")[0][:100] if text else "Unknown"  # 取第一行作为标题
                n = self.add_paper(arxiv_id, text, {"title": title})  # 添加到向量库
                if n > 0:
                    results.append(f"✅ {arxiv_id}: {n} chunks (\"{title}\")")  # 索引成功
                else:
                    results.append(f"⏭️  {arxiv_id}: already indexed")  # 已索引过，跳过
            except Exception as e:
                results.append(f"❌ {arxiv_id}: {e}")  # 索引失败

        return "\n".join(results)  # 返回多行进度报告

    def get_stats(self) -> str:
        """返回向量库统计：多少 chunk、多少篇论文。"""
        count = self.collection.count()  # 总 chunk 数
        papers = set()  # 用集合去重统计论文数
        if count > 0:
            all_meta = self.collection.get(include=["metadatas"])  # 获取所有元数据
            if all_meta["metadatas"]:
                papers = {m["arxiv_id"] for m in all_meta["metadatas"] if m}  # 从元数据中提取所有 arxiv_id
        return f"📚 {count} chunks from {len(papers)} papers indexed."

    def remove_paper(self, arxiv_id: str) -> int:
        """从向量库中删除某篇论文的所有 chunk。

        参数:
            arxiv_id: 要删除的论文 arXiv ID

        返回:
            删除的 chunk 数量"""
        existing = self.collection.get(where={"arxiv_id": arxiv_id})  # 查找该论文的所有 chunk
        if existing and existing["ids"]:  # 存在则删除
            self.collection.delete(ids=existing["ids"])  # 按 ID 批量删除
            return len(existing["ids"])
        return 0  # 没找到，返回 0


# ============================================================
# 单例模式：全局只创建一个 PaperVectorStore 实例
# 避免重复加载 embedding 模型（all-MiniLM-L6-v2 加载需要时间和内存）
# ============================================================
_store = None  # 模块级缓存

def get_vector_store() -> PaperVectorStore:
    """获取全局唯一的 PaperVectorStore 实例（延迟初始化）。"""
    global _store
    if _store is None:  # 第一次调用才创建
        _store = PaperVectorStore()
    return _store
