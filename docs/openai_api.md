## OpenAI-Compatible API

PAI-RAG支持使用OpenAI兼容的API访问，进而可以连接open-webui等常用的前端工具进行对话。

### 1、地址信息

1）ENDPOINT: 这里以PAI-EAS为例, EAS服务调用地址 `EAS_SERVICE_URL/v1`
2）API_KEY: 这里以PAI-EAS为例, `EAS_TOKEN`

### 2、网络搜索

- 调用地址：{ENDPOINT}/v1/chat/completions
- 请求方式：POST
- 请求 HEADERS
  Authorization: API_KEY # Eas调用token
- HTTP Body

```json
{
  "model": "default", # 模型名称，填default
  "messages": [
    {"role": "user","content": "你好"},
    {"role": "assistant","content": "你好，有什么能帮到您？"},
    {"role": "user", "content": "浙江省会是哪里"},
    {"role": "assistant", "content": "杭州是浙江的省会。"},
    {"role": "user","content": "有哪些好玩的"}
  ],
  "stream": true, # 是否流式
  "search_web": true, # 是否使用联网搜索，不传则搜索本地知识库
  "index_name": "default_index", # 索引名称，RAG场景使用，不传使用默认索引
}
```

### 3、示例

```python
from openai import OpenAI

##### API 配置 #####
openai_api_key = "API_KEY"
openai_api_base = "ENDPOINT"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


#### Chat ######
def chat():
    stream = True
    chat_completion = client.chat.completions.create(
        model="default",
        stream=stream,
        messages=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么能帮到您？"},
            {"role": "user", "content": "浙江省会是哪里"},
            {"role": "assistant", "content": "杭州是浙江的省会。"},
            {"role": "user", "content": "有哪些好玩的"},
        ],
        extra_body={
            "search_web": True,
        },
    )

    if stream:
        for chunk in chat_completion:
            print(chunk.choices[0].delta.content, end="")
    else:
        result = chat_completion.choices[0].message.content
        print(result)


chat()
```

- 返回 - 非流式输出

```json
{
  "id": "8df97a998f29485fb6b8f0fa1d65c52c",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "杭州有很多好玩的地方，以下是一些推荐的景点：\n\n1. **西湖** - 首批国家5A级旅游景区，中国十大风景名胜之一，以自然与人文景观著称。\n2. **西溪国家湿地公园** - 适合亲近自然，感受湿地的魅力。\n3. **灵隐寺** - 杭州最古老的寺庙之一，具有深厚的文化底蕴。\n4. **六和塔** - 杭州的标志性建筑之一，可以登塔俯瞰江景。\n5. **宋城** - 可以体验宋朝的历史文化。\n6. **雷峰塔** - “雷峰夕照”是杭州一大美景，可以欣赏夕阳下的西湖。\n7. **湘湖** - 湖光山色、古桥流水，适合休闲活动。\n8. **钱塘江大桥** - 不仅实用，还是观赏钱塘江壮丽景色的好地方。\n9. **京杭大运河** - 体验古代水运文化的绝佳地点。\n10. **太子湾公园** - 自然与人文景观结合的美丽公园。\n\n此外，还有其他一些免费景点也非常值得一去，比如杭州植物园、法喜寺、胡雪岩故居等。",
        "refusal": null,
        "role": "assistant",
        "audio": null,
        "function_call": null,
        "tool_calls": null
      }
    }
  ],
  "created": 1739450868,
  "model": "qwen-turbo",
  "object": "chat.completion",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "total_tokens": 0,
    "completion_tokens_details": null,
    "prompt_tokens_details": null
  }
}
```

- 返回 - 流式输出 (SSE格式)

Chunk 结构

```json
{
  "id": "7eb65e8cbc62428ca7ae22782addc4d1",
  "choices": [
    {
      "delta": {
        "content": "坊",
        "function_call": null,
        "refusal": null,
        "role": "assistant",
        "tool_calls": null
      },
      "finish_reason": null,
      "index": 240,
      "logprobs": null
    }
  ],
  "created": 1739451105,
  "model": "DeepSeek-R1-Distill-Qwen-32B",
  "object": "chat.completion.chunk",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": null
}
```
