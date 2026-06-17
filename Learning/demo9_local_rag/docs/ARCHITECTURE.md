# Demo 9 架构说明

## 核心流程

```text
本地文档
  -> 文本读取
  -> chunk 切分
  -> token 化
  -> TF-IDF 索引
  -> query 检索 top-k chunk
  -> 拼接 prompt
  -> LLM 或 placeholder 生成回答
  -> 输出 sources
```

## 为什么先用 TF-IDF

TF-IDF 不需要模型、不需要 API key、不需要向量数据库，适合用来学习 RAG 的工程骨架。它的缺点是语义理解弱，例如“静态搜索”和“Quiescence Search”如果文档里没有共同词，可能检索不到理想结果。

真实项目通常会替换为：

- Embedding 模型：把文本转成向量。
- 向量库：Qdrant、Milvus、pgvector、SQLite-vec 等。
- Rerank 模型：对初步召回结果重新排序。

## 模块职责

`build_chunks`

读取文档并切块。当前按段落累积到固定字符数，保留少量 overlap，避免跨段信息被完全切断。

`tokenize`

同时处理英文/代码 token 和中文字符。中文没有引入分词库，因此用单字和双字组合做轻量检索。

`retrieve`

把问题和文档 chunk 都转成 TF-IDF 向量，然后用 cosine similarity 排序。

`PlaceholderLlmClient`

不调用网络，只把检索命中的证据整理出来。它让 demo 在没有 key 的情况下也能跑通。

`OpenAICompatibleClient`

预留真实大模型调用，使用 `/v1/chat/completions` 风格接口。后续只需要配置 `api_key`、`base_url`、`model`。

## 真实项目注意点

1. 不要把 API key 写进代码或提交到仓库。
2. 文档入库前要考虑隐私、权限和数据来源。
3. 回答必须暴露引用来源，尤其是面向用户的知识库产品。
4. 对高风险场景要明确“资料不足”而不是让模型硬编。
5. 索引文件是生成物，文档变化后需要重新 build。
