"""
RAG System - Milestone 1 (离线版本)
支持在无法访问Hugging Face的环境中运行
"""

import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================================
# Configuration
# ============================================================================

DOCS_FOLDER = "/home/wutianhao/pycharm/RAG/txt"  # 存放81个txt文件的文件夹
LOG_FILE = "rag.log"
API_KEY = "sk-jvmcedkujodmpcxxnlhbnljxhvrhecjyaxxqixpgzpxdubzj"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2-Exp"

# 嵌入模型配置 - 支持多种方案
EMBEDDING_MODEL = "BAAI/bge-m3"
MODEL_CACHE_DIR = "/home/wutianhao/pycharm/RAG/model1"  # 本地模型缓存目录
USE_LOCAL_MODEL = True  # 优先使用本地模型

TOP_K = 5  # 检索前K个相关文档

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000

class RetrievedDoc(BaseModel):
    source: str
    text: str
    score: float

class RAGMetadata(BaseModel):
    retrieved_docs: List[RetrievedDoc]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatMessage(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"

class ChatResponse(BaseModel):
    choices: List[Choice]
    usage: Usage
    rag_metadata: RAGMetadata
    id: str
    object: str = "chat.completion"
    created: int
    model: str

# ============================================================================
# Document Processor
# ============================================================================

class DocumentProcessor:
    """处理和分块文档"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_documents(self, folder_path: str) -> List[Dict[str, str]]:
        """加载所有txt文件"""
        documents = []
        folder = Path(folder_path)

        if not folder.exists():
            logger.error(f"文档文件夹不存在: {folder_path}")
            return documents

        txt_files = list(folder.glob("*.txt"))
        logger.info(f"找到 {len(txt_files)} 个txt文件")

        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunks = self.split_text(content)

                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "source": file_path.name,
                            "text": chunk,
                            "chunk_id": i
                        })

                logger.info(f"加载文件: {file_path.name}, 分块数: {len(chunks)}")

            except Exception as e:
                logger.error(f"加载文件失败 {file_path.name}: {str(e)}")

        logger.info(f"总共加载 {len(documents)} 个文档块")
        return documents

    def split_text(self, text: str) -> List[str]:
        """将文本分块"""
        chunks = []
        text = text.strip()

        if len(text) == 0:
            return chunks

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap

        return chunks

# ============================================================================
# Vector Store
# ============================================================================

class VectorStore:
    """向量存储和检索 - 支持离线模式"""

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        logger.info(f"准备加载嵌入模型: {embedding_model_name}")
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        self.documents = []
        self.index = None
        self.dimension = None

    def _load_embedding_model(self, model_name: str):
        """加载嵌入模型 - 支持多种方式"""

        # 设置离线模式环境变量
        os.environ['HF_HUB_OFFLINE'] = '1' if USE_LOCAL_MODEL else '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '1' if USE_LOCAL_MODEL else '0'

        try:
            # 方法1: 尝试从本地缓存加载
            logger.info("尝试从本地缓存加载模型...")
            model = SentenceTransformer(
                model_name,
                cache_folder=MODEL_CACHE_DIR,
                device="cuda" if self._check_cuda() else "cpu"
            )
            logger.info(f"✓ 成功从本地加载模型: {model_name}")
            return model

        except Exception as e1:
            logger.warning(f"从本地加载失败: {str(e1)}")

            try:
                # 方法2: 尝试联网下载
                logger.info("尝试联网下载模型...")
                os.environ['HF_HUB_OFFLINE'] = '0'
                model = SentenceTransformer(
                    model_name,
                    cache_folder=MODEL_CACHE_DIR,
                    device="cuda" if self._check_cuda() else "cpu"
                )
                logger.info(f"✓ 成功下载并加载模型: {model_name}")
                return model

            except Exception as e2:
                logger.error(f"联网下载也失败: {str(e2)}")

                # 方法3: 使用备用的简单模型
                logger.warning("⚠️ 无法加载sentence-transformers模型，使用简单TF-IDF替代")
                return self._create_simple_embedder()

    def _check_cuda(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _create_simple_embedder(self):
        """创建简单的TF-IDF嵌入器作为备用方案"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        class SimpleTFIDFEmbedder:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=384)
                self.is_fitted = False

            def encode(self, texts, show_progress_bar=False):
                if not self.is_fitted:
                    vectors = self.vectorizer.fit_transform(texts)
                    self.is_fitted = True
                else:
                    vectors = self.vectorizer.transform(texts)
                return vectors.toarray().astype('float32')

        logger.info("✓ 创建简单TF-IDF嵌入器")
        return SimpleTFIDFEmbedder()

    def build_index(self, documents: List[Dict[str, str]]):
        """构建向量索引"""
        logger.info("开始构建向量索引...")
        self.documents = documents

        if len(documents) == 0:
            logger.warning("没有文档可以索引")
            return

        try:
            # 生成嵌入向量
            texts = [doc["text"] for doc in documents]
            logger.info(f"正在为 {len(texts)} 个文档块生成嵌入向量...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

            # 确保embeddings是numpy数组
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            embeddings = embeddings.astype('float32')

            # 构建FAISS索引
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)  # 使用内积相似度

            # 归一化向量（使内积等价于余弦相似度）
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

            logger.info(f"✓ 向量索引构建完成，维度: {self.dimension}, 文档数: {len(documents)}")

        except Exception as e:
            logger.error(f"构建索引失败: {str(e)}")
            raise

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if self.index is None or len(self.documents) == 0:
            logger.warning("索引未构建或文档为空")
            return []

        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode([query])

            # 确保是numpy数组
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)

            # 搜索
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    results.append({
                        "source": self.documents[idx]["source"],
                        "text": self.documents[idx]["text"],
                        "score": float(score)
                    })

            logger.info(f"检索到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []

# ============================================================================
# RAG Engine
# ============================================================================

class RAGEngine:
    """RAG核心引擎"""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()

    def initialize(self, docs_folder: str):
        """初始化RAG系统"""
        logger.info("=" * 80)
        logger.info("RAG系统启动")
        logger.info("=" * 80)

        # 加载文档
        documents = self.doc_processor.load_documents(docs_folder)

        if len(documents) == 0:
            logger.error("未加载到任何文档，请检查文档文件夹")
            raise ValueError("No documents loaded")

        # 构建索引
        self.vector_store.build_index(documents)

        logger.info("RAG系统初始化完成")
        logger.info("=" * 80)

    def generate_response(self, messages: List[Dict[str, str]],
                         temperature: float = 0.7,
                         max_tokens: int = 2000) -> Dict[str, Any]:
        """生成RAG响应"""
        start_time = datetime.now()

        # 获取最后一个用户消息作为查询
        query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                query = msg["content"]
                break

        logger.info(f"收到查询: {query[:100]}...")

        # 检索相关文档
        retrieved_docs = self.vector_store.search(query, top_k=TOP_K)

        # 构建增强的prompt
        context = "\n\n".join([
            f"[文档 {i+1} - {doc['source']}]\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        augmented_messages = messages.copy()
        if context:
            system_message = {
                "role": "system",
                "content": f"你是一个专业的助手。请基于以下提供的文档内容来回答用户的问题。如果文档中没有相关信息，请说明并基于你的知识给出回答。\n\n相关文档:\n{context}"
            }
            augmented_messages.insert(0, system_message)

        # 调用LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=augmented_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 计算耗时
            elapsed_time = (datetime.now() - start_time).total_seconds()

            # 构建响应
            result = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "rag_metadata": {
                    "retrieved_docs": [
                        {
                            "source": doc["source"],
                            "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                            "score": doc["score"]
                        }
                        for doc in retrieved_docs
                    ]
                },
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": self.model_name
            }

            logger.info(f"响应生成完成，耗时: {elapsed_time:.2f}秒")
            logger.info(f"Token使用: {result['usage']}")

            return result

        except Exception as e:
            logger.error(f"生成响应失败: {str(e)}")
            raise

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="RAG API", version="1.0.0")
rag_engine = RAGEngine(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)

@app.on_event("startup")
async def startup_event():
    """启动时初始化RAG系统"""
    rag_engine.initialize(DOCS_FOLDER)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI兼容的聊天接口"""
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        response = rag_engine.generate_response(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "documents_loaded": len(rag_engine.vector_store.documents)}

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")