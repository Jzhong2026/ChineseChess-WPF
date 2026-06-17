# Demo 9: 本地知识库 RAG 助手

这个 demo 用最少依赖演示一个完整 RAG 流程：

1. 读取本地 Markdown/TXT/C#/Python 文档。
2. 把文档切成 chunk。
3. 用本地 TF-IDF 做检索。
4. 把命中的上下文交给 LLM。
5. 没配置大模型时，用 placeholder 回答展示检索证据。

它适合学习 AI 应用里最常见的一条链路：**资料入库 -> 检索 -> 拼上下文 -> 模型回答 -> 引用来源**。

## 目录

```text
Learning/demo9_local_rag/
  rag_demo.py                 # 主脚本
  ai_config.example.json      # OpenAI-compatible 配置模板
  requirements.txt            # 当前无第三方依赖
  sample_docs/                # 示例知识库
  docs/ARCHITECTURE.md        # 设计说明
```

## 运行

先建立索引：

```powershell
python Learning\demo9_local_rag\rag_demo.py build
```

只看检索结果：

```powershell
python Learning\demo9_local_rag\rag_demo.py search "QSearch 解决什么问题"
```

问答：

```powershell
python Learning\demo9_local_rag\rag_demo.py ask "RAG 的流程是什么"
```

未配置大模型时，`ask` 会输出 placeholder 回答，并列出命中的引用来源。

## 接入大模型

生成本地配置：

```powershell
python Learning\demo9_local_rag\rag_demo.py init-config
```

然后编辑 `Learning\demo9_local_rag\ai_config.json`：

```json
{
  "provider": "openai-compatible",
  "api_key": "",
  "base_url": "",
  "model": "",
  "temperature": 0.2,
  "timeout_seconds": 60
}
```

`base_url` 支持这几种形式：

- `https://api.example.com`
- `https://api.example.com/v1`
- `https://api.example.com/v1/chat/completions`

配置完成后运行：

```powershell
python Learning\demo9_local_rag\rag_demo.py ask "RAG 和普通聊天有什么区别" --provider openai-compatible
```

真实 `ai_config.json` 已加入 `.gitignore`，不要提交 API key。

## 换成自己的资料

可以指定文档目录：

```powershell
python Learning\demo9_local_rag\rag_demo.py build --docs docs
python Learning\demo9_local_rag\rag_demo.py ask "这个项目的训练流程是什么"
```

支持文件类型：

- `.md`
- `.txt`
- `.cs`
- `.py`

## 学到什么

1. RAG 不是让模型“记住”资料，而是在回答前先检索相关资料。
2. chunk 太大时上下文不精准，太小时容易丢失语义。
3. 检索结果要带来源，否则用户无法判断答案可信度。
4. LLM 客户端应通过接口隔离，方便以后切换 OpenAI、DeepSeek、通义、智谱或本地模型。
5. placeholder 很有用：可以先验证应用流程，再接真实模型。

## 后续可扩展

- 把 TF-IDF 换成 Embedding + 向量库。
- 给每个回答加“引用片段高亮”。
- 做一个 WPF 前端：左侧管理文档，右侧聊天问答。
- 增加增量索引，只更新变更过的文件。
- 加入 Tool Calling，让模型能主动执行 `build/search` 等工具。
