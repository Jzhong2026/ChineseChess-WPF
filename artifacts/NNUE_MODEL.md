# 象棋 NNUE ONNX 模型来源与复现

应用文件名：`xiangqi_nnue_value.onnx`，在WPF中选择“ONNX 加强”使用。2026-11-13已完成正式40局：对当前传统困难Level4，双方每步1200ms、20开局交换红黑，36胜0负4和、得分95%；逐手审计与应用本地测试通过，详见[开发日志](../docs/devlog/2026-11-13.md)。原CNN模型保留。

## 来源

- 原始网络：`xiangqi-c07e94a5c7cb.nnue`，Pikafish developers 为 Fairy-Stockfish 发布的象棋网络。
- [官方网络目录](https://fairy-stockfish.github.io/nnue/)；[对应发布页](https://github.com/fairy-stockfish/Fairy-Stockfish-NNUE/releases/tag/xiangqi-c07e94a5c7cb)。
- 原始权重 SHA256：`c07e94a5c7cbeae443ed79a8fa412875d833a7f8e04333815e39729c59d52e11`。
- 转换后 ONNX SHA256：`d185b8af242a29201d3e0eb610b68ef20e7de16ff551217c383081a48b780762`。
- 采用公开预训练权重转换；本机CUDA训练用于其他候选对照，所有训练和失败结果均保留在开发日志。
- 原作者NNUE-License单独说明其为Fairy-Stockfish训练的网络使用CC0；随模型保留该许可原文。其他Pikafish专用权重的条件不应与本文件混淆。

## 运行与格式

应用使用本项目的通用alpha-beta搜索，由ONNX提供静态局面分数。外部参考程序用于离线转换核验，应用运行不需要它。

模型为自包含ONNX，使用标准ORT CPU算子。输入`us`、`them`各为`int64[batch,32]`，保存双方HalfKAv2Variants活跃特征；`bucket`为`int64[batch]`，取值0–7。10530是零填充特征。输出`value_cp`为当前行棋方视角的分值。metadata中的编码标识为`xiangqi-halfka-v2-10530-v1`。

黑方仅反射rank。将的位置决定九个宫内桶，棋子类型顺序与本项目枚举不同。转换保留量化激活、截断与PSQT/网络分桶，不能套用普通CNN或MLP编码。

## 复现

先准备有明确来源的原始权重及对应离线参考程序。使用安装有PyTorch、ONNX、ONNX Runtime的Python环境，从仓库根执行：

```powershell
python tools/nnue/convert_fairy.py `
  --input <xiangqi-c07e94a5c7cb.nnue> `
  --reference-engine <fairy-reference.exe> `
  --samples <teacher-samples.jsonl> `
  --output-dir <workspace-tmp/new-export>
```

输出目录必须为空。工具校验具体权重hash和完整文件尺寸，核对原生参考评价，再检查ONNX数值。当前验证环境PyTorch2.11.0、Python ORT1.31.1；应用ORT1.19.2也已独立进行C#数值检查。外部预训练语料不可得，不宣称验收开局与该语料完全无重叠。

完整对局需另外运行AiMatchRunner；数值一致性、开发赛成绩与正式40局门槛是三种不同证据。
