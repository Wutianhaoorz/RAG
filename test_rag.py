import requests
import json

# 测试地址
BASE_URL = "http://localhost:8000"

print("=" * 60)
print("测试 RAG API")
print("=" * 60)

# 1. 测试健康检查
print("\n1. 测试健康检查...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
except Exception as e:
    print(f"❌ 错误: {e}")

# 2. 测试聊天API
print("\n2. 测试聊天API...")
try:
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "你使用的是什么LLM模型？"}
            ],
            "model": "deepseek",
            "temperature": 0.7,
            "max_tokens": 500
        },
        timeout=60  # 60秒超时
    )

    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()

        # 打印回答
        print("\n" + "=" * 60)
        print("AI回答:")
        print("=" * 60)
        print(result['choices'][0]['message']['content'])

        # 打印检索到的文档
        print("\n" + "=" * 60)
        print("检索到的文档:")
        print("=" * 60)
        for i, doc in enumerate(result['rag_metadata']['retrieved_docs'], 1):
            print(f"\n文档 {i}:")
            print(f"  来源: {doc['source']}")
            print(f"  分数: {doc['score']:.4f}")
            print(f"  内容: {doc['text'][:100]}...")

        # 打印token使用情况
        print("\n" + "=" * 60)
        print("Token使用情况:")
        print("=" * 60)
        print(f"  Prompt tokens: {result['usage']['prompt_tokens']}")
        print(f"  Completion tokens: {result['usage']['completion_tokens']}")
        print(f"  Total tokens: {result['usage']['total_tokens']}")
    else:
        print(f"❌ 请求失败: {response.text}")

except Exception as e:
    print(f"❌ 错误: {e}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)