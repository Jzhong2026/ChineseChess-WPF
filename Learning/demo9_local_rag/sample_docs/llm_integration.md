# 大模型接入方式

AI 应用里不要把大模型调用散落在业务代码中，建议抽象成统一客户端。客户端只暴露 chat、embedding 或 tool calling 等能力，业务层不关心供应商细节。

本 demo 预留了 OpenAI-compatible chat completions 接口。很多模型服务都支持类似格式，只需要配置 api_key、base_url 和 model。

推荐配置项：

- api_key：访问模型服务的密钥。
- base_url：模型服务地址。
- model：使用的模型名称。
- temperature：控制回答随机性，知识库问答通常设置得较低。
- timeout_seconds：网络请求超时时间。

如果暂时没有 API key，可以先使用 placeholder。placeholder 不负责真正推理，只负责展示检索命中的上下文，帮助开发者先验证 RAG 流程。

## Prompt 约束

知识库问答的 system prompt 应要求模型只根据上下文回答。如果上下文不足，模型应该明确说明缺少什么资料，而不是编造答案。
