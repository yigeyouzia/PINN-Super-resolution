---
title: 自维护 PINN 源码与开发入口
summary: train.py 模块化后的源码职责、导入方式、训练与 checkpoint API，以及已验证范围和后续开发顺序。
type: directory-index
status: implemented-locally-verified
created: 2026-10-19
updated: 2026-10-19
papers:
  - "An airflow velocity field reconstruction method with sparse or incomplete data using physics-informed neural network; DOI: 10.1016/j.jobe.2024.109231"
github_repositories:
  - https://github.com/yigeyouzia/PINN-Super-resolution
  - https://github.com/wanghuan10/PINN-Super-resolution
---

# 自维护源码：pinn_sr

> 总结日期：2026-10-19。
> 状态：训练模块已实现，Mac CPU 回归与冒烟通过；Linux CUDA 待验证。
> 进度交接：[开发规划与交付进度](./开发规划与交付进度.md)。
> 作者 [pinn/](../pinn/) 是原始实现与对照来源，不修改；这里维护后续开发版本。

## 1. 从哪里开始

外部训练命令继续使用根 [train.py](../train.py)，参数和完整 Mac/Linux 指令见[参数说明](../docs/experiments/02_configurable_training.md)。根入口仅组织调用，并保留 `from train import restore_checkpoint` 用法。

本地采用 `src.pinn_sr` 导入路径，不需要安装项目包或手动修改 PYTHONPATH；从仓库根目录导入，或使用根训练脚本的绝对路径执行。不要直接执行 `python src/pinn_sr/pinn_train.py`，该文件是模块，不是独立 CLI。

```text
train.py → cli.py → 配置 / 数据 / 运行环境检查
                           ↓
                     pinn_train.py
                      ↙         ↘
              pinn.py/network.py   实验日志与 checkpoint
                                         ↓
                              独立恢复模型（不创建优化器）
```

| 模块 | 职责 | 后续修改位置 |
| --- | --- | --- |
| [config.py](./pinn_sr/config.py) | CLI 参数、默认值、方法契约及续训比较字段 | 新增参数先在这里声明 |
| [cli.py](./pinn_sr/cli.py) | 检查、来源记录、续训准入与训练调用 | 新入口不要复制整个流程 |
| [runtime.py](./pinn_sr/runtime.py) | 设备、精度运行支持、随机种子与环境信息 | CPU/CUDA 策略 |
| [data.py](./pinn_sr/data.py) | CSV 数据校验与可恢复配点采样 | 新数据适配器应独立添加，不污染作者目录 |
| [network.py](./pinn_sr/network.py) | Tanh/Xavier MLP 与坐标归一化 buffer | 后续网络结构 |
| [pinn.py](./pinn_sr/pinn.py) | 三维稳态预测、损失项与自动微分残差 | 后续 PDE/损失修改 |
| [pinn_train.py](./pinn_sr/pinn_train.py) | 优化器、训练循环、采样/随机状态恢复与保存调度 | 优化与训练机制 |
| [checkpoint.py](./pinn_sr/checkpoint.py) | 状态读写、模型恢复、严格续训/旧版本迁移 | 版本兼容必须附测试 |
| [experiment.py](./pinn_sr/experiment.py) | 项目根路径、来源哈希、唯一目录与防覆盖 | 实验产物规则 |
| [metrics.py](./pinn_sr/metrics.py) | 纯数值指标归约、分批训练点拟合评价 | 后续独立评估复用 |

## 2. 保留的方法与工程变化

保留当前基线的层次、参数注册顺序、Tanh/Xavier、坐标归一化、逐点绝对相对误差求和、三维物理残差、可学习系数初始化、Adam 和默认 5:1 权重。alpha 虽未参与损失，仍保留以兼容旧状态。

工程变化：

- 不再动态导入/绑定作者方法，也不在正式代码中替换 sys.modules；作者对照加载器仅放在 tests/。
- 模型不创建 Adam，优化器由训练模块的 make_optimizer 管理。
- 模型权重与归一化 buffer 的 v1 键保持不变；模型恢复采用 strict=True。
- 相对 data/、runs/ 路径仍相对于项目根目录，不是 src/ 或当前工作目录。
- 输出增加 src/ 保护，继续独占创建运行目录和文件。
- 依然只有稳态空间任务，没有时间输入、企业 DAT、ConFIG 或自动图片生成。

## 3. 常用命令

下面在项目根目录、已有 torch/numpy 环境中执行：

```bash
python -B -m unittest discover -s tests -v
python -B train.py --check-only --device cpu
python -B train.py --device cpu --num-threads 1 --steps 20 --run-tag refactor-smoke
```

Linux 在已有环境检查通过后，将训练设备指定为 `--device cuda`。这些命令不安装环境、不修改系统 CUDA。新版本尚未在 Linux GPU 上运行，旧版本服务器训练成功不等于新版本通过验证。

## 4. 为 evaluate.py 提供的恢复接口

```python
from pathlib import Path
import torch
from src.pinn_sr.checkpoint import restore_checkpoint

# 替换成已有的真实路径，不能照搬占位符。
checkpoint = Path("runs/实际运行目录/checkpoints/step_00500000.pt")
model, state = restore_checkpoint(checkpoint, device="cpu")

# xyz 为 N×3 的物理坐标；内部使用 checkpoint 中训练时的归一化边界。
inputs = torch.as_tensor(xyz, dtype=getattr(torch, state["config"]["dtype"]))
with torch.no_grad():
    uvwp = model.net(inputs)
```

该代码是接口示例，xyz 与 checkpoint 路径需由调用方提供；尚未实现 evaluate.py。速度推理无需优化器，PDE 残差计算则需要启用坐标自动微分，不能套用 no_grad。

## 5. checkpoint 与续训约束

- 继续使用 `pinn-sr-state-v1` 状态格式，读取时始终 weights_only=True；作者完整对象模型不在此接口支持范围内。
- 新训练额外记录 method_contract 和所有维护源码的 SHA-256。
- 新版同源码续训仍校验网络、精度、优化器、学习率、损失权重、seed、采样和数据内容；源码或方法契约变化不能静默续训。
- 旧版 v1 可直接推理；续训需重复原训练参数，并额外给出 `--allow-legacy-resume`。
- 此选项只允许 checkpoint.py 中列出的、已核验的重构前源码哈希组合，不是跳过兼容检查的开关。未知版本应先核验来源，不能强行放行。
- 迁移后始终创建新目录，并在 config.json 的 resume_compatibility 中记录旧来源。
- 服务器实际 500000 步权重尚未在本机读取；若源码哈希与已核验组合不同，可以先推理验证，再决定是否补充迁移配置。

## 6. 本次验证与限制

| 验证 | 结果 |
| --- | --- |
| Mac CPU 自动回归 | 24 项通过；包含原 10 项测试意图 |
| 与作者实现对照 | 固定 float32 输入的前向、损失、梯度逐值一致 |
| 与重构前实际 checkpoint 对照 | float32/float64 × sequential/random；4 步从头训练及旧 2 步权重续训到 4 步的模型、优化器、采样器和 Torch RNG 均逐值一致 |
| 作者 data/5 | 10×100、20 步、物理 batch=128、CPU 单线程，保存/恢复通过 |
| 输出与原始资料 | 测试实验写到系统临时目录；作者代码、数据、模型无修改 |
| Linux GPU / 服务器实际权重 | 待验证，未访问服务器 |
| 论文精度、独立测试集、图片 | 本次不验证/不实现 |

开发验证环境：macOS 15.3 arm64、Python 3.13.5、PyTorch 2.9.0、NumPy 2.3.4，沿用已有环境，未安装新依赖。临时实验不作为长期归档或论文结果；可复算的常规测试在 tests/。

## 7. 下一步

先在服务器做新版本 CUDA 冒烟，并核对已有权重的恢复；随后按规划实现 evaluate.py，再做无界面可视化。不要为获得图片重新训练已有 500000 步实验。
