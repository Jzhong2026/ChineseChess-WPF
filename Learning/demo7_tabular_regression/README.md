# Demo 7: PyTorch 表格回归

这是一个典型的结构化数据项目：用房屋面积、房间数、房龄、距离、市学区评分等特征预测房价。数据是脚本内部合成的，不需要下载外部数据。

## 典型应用

- 房价、销量、收入、耗时等连续数值预测
- 风控评分、用户价值、设备寿命预测
- 游戏或棋类项目里的评估分数拟合

## 运行

```powershell
python Learning\demo7_tabular_regression\tabular_regression_demo.py train
```

评估：

```powershell
python Learning\demo7_tabular_regression\tabular_regression_demo.py eval --model artifacts\tabular_house_price.pt
```

单条预测：

```powershell
python Learning\demo7_tabular_regression\tabular_regression_demo.py predict --area 96 --rooms 3 --age 8 --distance 6 --school-score 7
```

## 学到什么

1. 表格特征要做标准化。
2. 回归任务常用 MSE loss 训练，用 MAE/RMSE 评估。
3. 模型保存时不只保存权重，也要保存归一化参数。
4. MLP 是表格任务最基础的 PyTorch baseline。
