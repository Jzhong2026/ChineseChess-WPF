# Demo 6: PyTorch 手写数字识别

这个 demo 用 Python + PyTorch 做一个 MNIST 手写数字识别初版。它和前面的神经网络 demo 一样偏教学用途：代码少、结构直接、能训练、能评估、能保存模型，也能对一张图片做预测。

## 复杂度评估

整体复杂度：低到中等，适合作为 PyTorch 入门 demo。

| 模块 | 难度 | 说明 |
|---|---:|---|
| 环境准备 | 低 | 安装 `torch`、`torchvision`、`Pillow` 即可 |
| 数据集 | 低 | 直接使用 `torchvision.datasets.MNIST` 下载和加载 |
| 模型 | 低 | 两层卷积 + 两层全连接，结构很小 |
| 训练 | 中低 | 需要理解 batch、loss、optimizer、epoch |
| 评估 | 低 | 计算 cross entropy loss 和 accuracy |
| 单图预测 | 中低 | 需要把外部图片转成 28x28 灰度张量 |
| 工程化 | 中 | 后续若要 UI、Web API、ONNX、部署，复杂度会上升 |

初版工作量通常是半天以内；如果加 UI 绘图板、模型导出、WPF 调用或 Web 服务，可以扩展到 1-3 天。

## 运行

在项目根目录执行：

```powershell
python Learning\demo6_mnist_pytorch\mnist_pytorch_demo.py train
```

默认配置会：

- 下载 MNIST 到 `data/mnist`
- 使用 12000 条训练样本、2000 条测试样本
- 训练 1 个 epoch
- 保存模型到 `artifacts/mnist_cnn.pt`

评估已保存模型：

```powershell
python Learning\demo6_mnist_pytorch\mnist_pytorch_demo.py eval --model artifacts\mnist_cnn.pt
```

使用自己的图片预测：

```powershell
python Learning\demo6_mnist_pytorch\mnist_pytorch_demo.py predict --model artifacts\mnist_cnn.pt --image path\to\digit.png
```

## 提高准确率

初版为了快，默认只使用部分数据。想要更高准确率可以跑完整 MNIST：

```powershell
python Learning\demo6_mnist_pytorch\mnist_pytorch_demo.py train --train-limit 0 --test-limit 0 --epochs 3
```

一般这个小 CNN 在 MNIST 上很容易超过 98% 准确率。

## 和象棋项目的关系

这个 demo 的重点不是数字识别本身，而是把 PyTorch 训练流程拆小：

1. `Dataset/DataLoader` 负责喂数据。
2. `nn.Module` 定义模型结构。
3. `cross_entropy` 定义分类损失。
4. `optimizer.step()` 更新参数。
5. `torch.save()` 保存权重。

这些概念可以迁移到当前象棋项目的 policy/value 网络训练里。
