# 开发实验与学习日志

本目录按用户要求记录开发中的问题、尝试、失败、修复和收获。原始数据与临时脚本放在工作区根 `tmp/`，日志正文保留在仓库。失败实验不能从记录中删除，未完成实验不能写成成功。

- [2026-11-12：ONNX 棋力、CUDA 蒸馏、MCTS 与正式对弈](2026-11-12.md)
- [2026-11-13：ONNX价值搜索、预训练NNUE转换与40局95%验收](2026-11-13.md)

## 每次实验要留下什么

1. **问题与假设**：实际观察到了什么；猜测的原因是什么，如何区分。
2. **可复现输入**：代码/程序集/模型/数据 hash，随机种子、预算、依赖版本和命令。
3. **尝试与结果**：成功、失败、中止都写，原始输出存到同一任务的 tmp 子目录。
4. **解释与限制**：区分编译/单测、数值一致性、验证集指标、实战胜负；多项同时变化不能归因给其中一项。
5. **下一步**：下一次实验要排除哪个原因；达标前保留可回退的基线。

## 本次学到的关键区别

- “可以用GPU训练”不等于模型已学好；显存没有占满不等于训练失败。要同时记录吞吐、验证集表现和实际棋力。
- “验证准确率更高”不等于实战更强。教师正常对局与模型自身失误轨迹的分布不同，需要实际败局复盘。
- “搜索次数更多”不等于选择更好。错误的价值、策略先验或终局处理会被搜索继续放大。
- 神经网络负责策略与局势判断，有限战术延伸负责检查反吃等短期变化。两者的贡献应分别验证，不能把混合系统的成绩全部写成模型本身的提升。
- 优化要先测瓶颈：本次大头曾是C#逐元素Tensor多维索引，而非ONNX矩阵计算。优化后要逐输出验证语义，再测真实对局。
- 同一开局执红/执黑配对、双方相同时间、固定传统对手二进制、预先冻结模型和开局，才能形成可解释的比较。
- 相同随机种子并不能保证按时限搜索得到完全相同的着法：机器负载影响可完成的模拟数。应保留实际耗时、模拟数和搜索深度。
- Python pytest的 `--basetemp` 只控制临时测试目录，不控制 `.pytest_cache`；本次后续命令用 `-p no:cacheprovider`。
- 小样本蒸馏和公开大规模预训练权重是两条路线。本次胜出的是后者转换为ONNX配合通用搜索；本机CUDA训练失败不能隐去，也不能把组合胜率归为纯网络胜率。单步后继选择用于观察去掉深搜索后的变化。
- 模型转换先校验权重格式/许可，再核对跨语言特征、量化算术与参考输出；浮点能精确表示一定范围的整数，但需要先证明中间值范围。数值检查通过后仍必须实战。

## 复现入口

以下命令从仓库根执行；`<tmp>` 指当前工作区根的 `tmp/onnx-goal-20261112`，不是系统TEMP。每次实验使用新的输出路径。

```powershell
# 离线教师：可执行文件及NNUE须由使用者另行准备，项目不打包教师。
dotnet run --project ChineseChess/ChineseChess.TeacherData -c Release -- `
  <teacher.exe> <tmp>/new-teacher.jsonl 256 20261112 20000 220

# GPU训练：输入必须已停止写入；输出是候选，不能直接视为通过验收。
<tmp>/venv/Scripts/python.exe train_teacher.py `
  --input <tmp>/teacher-pilot.jsonl <tmp>/teacher-expanded-first512.jsonl `
  --warm-start artifacts/cnn_policy_value.onnx --output-dir <tmp>/new-training `
  --epochs 50 --batch-size 512 --material-skip --soft-policy

# 成对对弈：opening文件取TeacherData中MoveIndex=8的History，每开局两局。
dotnet run --project ChineseChess/AiMatchRunner -c Release -- --arena `
  mcts 40 <tmp>/new-arena <candidate.onnx> <openings.jsonl> 3000 1200 <settings.json>

# 本次胜出的ONNX价值+alpha-beta模式，同样需独立新输出目录。
dotnet run --project ChineseChess/AiMatchRunner -c Release -- --arena `
  nnue 40 <tmp>/new-nnue-arena artifacts/xiangqi_nnue_value.onnx <openings.jsonl> 3000 1200
```

正式验收实际使用已复制冻结的DLL，而不是边开发边重编译的 `dotnet run`。具体路径、参数、hash、失败修正和结果均以当日日志与冻结元数据为准。
