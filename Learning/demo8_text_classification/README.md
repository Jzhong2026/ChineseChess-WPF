# Demo 8: PyTorch 文本分类

这是一个典型的 NLP 入门项目：把一句短文本判断为正向或负向。为了让 demo 可离线运行，语料直接写在脚本里，模型使用 `Embedding + BiGRU + Linear`。

## 典型应用

- 评论情感分类
- 工单/消息意图分类
- 文档标签分类
- 棋类项目里的日志、局面说明、训练样本备注分类

## 运行

```powershell
python Learning\demo8_text_classification\text_classification_demo.py train
```

评估：

```powershell
python Learning\demo8_text_classification\text_classification_demo.py eval --model artifacts\text_sentiment_gru.pt
```

单句预测：

```powershell
python Learning\demo8_text_classification\text_classification_demo.py predict --text "这个项目体验很好"
```

## 学到什么

1. 文本要先分词并映射成 token id。
2. 不同长度的句子需要 padding，并把真实长度传给 RNN。
3. 文本分类常用 cross entropy loss。
4. 保存模型时也要保存词表，否则预测阶段无法复现 token id。

这个 demo 的语料很小，只适合理解流程。真实项目通常会换成更大的数据集，并使用 Transformer、BERT 或其它预训练模型。
