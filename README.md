# RAG 系统 - Milestone 1

基于检索增强生成（RAG）的问答系统，提供OpenAI兼容的API接口。

## 📋 功能特性

- ✅ **文档处理**：自动加载和分块处理81个txt文件
- ✅ **向量检索**：使用FAISS进行高效相似度搜索
- ✅ **OpenAI兼容API**：标准的OpenAI格式接口
- ✅ **并发支持**：支持多个API请求同时调用
- ✅ **详细日志**：完整的请求和错误日志记录
- ✅ **RAG元数据**：返回检索到的文档和相关性分数

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备文档

将81个txt文件放入 `docs` 文件夹中（如果文件夹不存在则创建）：

```
project/
├── docs/
│   ├── paper_01.txt
│   ├── paper_02.txt
│   └── ...
├── rag_system.py
└── requirements.txt
```

### 3. 启动服务

```bash
python rag_system.py
```

服务将在 `http://localhost:8000` 启动。

## 📡 API 使用

### 聊天接口

**请求示例：**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is ignition delay?"}
    ],
    "model": "gpt-4"
  }'
```

**使用Python：**

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is ignition delay?"}
        ],
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000
    }
)

result = response.json()
print(result['choices'][0]['message']['content'])
print(f"检索到 {len(result['rag_metadata']['retrieved_docs'])} 个相关文档")
```

**使用OpenAI SDK：**

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # 本地服务不需要真实API key
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What is ignition delay?"}
    ]
)

print(response.choices[0].message.content)
```

### 响应格式

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Ignition delay is the time between..."
    },
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 150,
    "total_tokens": 1350
  },
  "rag_metadata": {
    "retrieved_docs": [
      {
        "source": "paper_01.txt",
        "text": "The ignition delay time is defined as...",
        "score": 0.89
      },
      {
        "source": "paper_23.txt", 
        "text": "In combustion systems, the delay period...",
        "score": 0.76
      }
    ]
  },
  "id": "chatcmpl-1697876543.123",
  "object": "chat.completion",
  "created": 1697876543,
  "model": "deepseek-ai/DeepSeek-V3.2-Exp"
}
```

### 健康检查

```bash
curl http://localhost:8000/health
```

## 📊 日志文件

日志文件 `rag.log` 记录所有系统活动：

- **启动日志**：文档加载统计
- **请求日志**：每次API调用的详细信息
- **错误日志**：异常和错误信息

## ⚙️ 配置说明

在 `rag_system.py` 中可以调整以下参数：

```python
DOCS_FOLDER = "./docs"           # 文档文件夹路径
TOP_K = 5                         # 检索文档数量
chunk_size = 500                  # 文档分块大小
overlap = 50                      # 分块重叠大小
EMBEDDING_MODEL = "..."           # 嵌入模型
```

## 🔧 技术架构

- **Web框架**：FastAPI（支持异步和并发）
- **向量检索**：FAISS + sentence-transformers
- **LLM调用**：OpenAI SDK（使用SiliconFlow API）
- **嵌入模型**：sentence-transformers/all-MiniLM-L6-v2

## 📝 工作流程

1. **文档加载**：读取所有txt文件并分块
2. **向量化**：使用sentence-transformers生成嵌入向量
3. **索引构建**：使用FAISS构建向量索引
4. **查询处理**：
   - 接收用户问题
   - 向量化查询
   - 检索Top-K相关文档
   - 构建增强prompt
   - 调用LLM生成回答
5. **返回结果**：包含答案、token统计和RAG元数据

## 🐛 故障排除

### 文档未加载

- 确保 `docs` 文件夹存在
- 检查txt文件编码（应为UTF-8）
- 查看 `rag.log` 中的加载日志

### API调用失败

- 确认API key和base_url正确
- 检查网络连接
- 查看错误日志

### 内存不足

- 减少 `TOP_K` 值
- 增加 `chunk_size` 以减少文档块数量
- 考虑使用更小的嵌入模型

## 📈 下一步（Milestone 2）

- 添加更详细的性能统计
- 支持更多文档格式
- 实现文档缓存机制
- 优化检索算法
- 添加重排序（reranking）

## 📞 支持

如有问题，请查看日志文件或提交issue。
