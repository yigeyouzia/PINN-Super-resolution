---
title: 可配置训练入口：参数说明与 Mac/Linux 指令
summary: train.py 保留作者原数据损失和三维物理残差，新增显式参数、独立运行目录、状态 checkpoint、日志与安全续训。提供 Mac CPU 和 Linux CUDA 测试及完整训练命令，标明实际验证范围。
type: repo-note
status: partially-reproduced
created: 2026-10-14
updated: 2026-10-19
papers:
  - "An airflow velocity field reconstruction method with sparse or incomplete data using physics-informed neural network; DOI: 10.1016/j.jobe.2024.109231"
github_repositories:
  - https://github.com/yigeyouzia/PINN-Super-resolution
  - https://github.com/wanghuan10/PINN-Super-resolution
data:
  - 作者公开 CSV；Mac CPU 工程冒烟测试已运行，论文指标尚未复现
---

# 可配置训练入口：参数与执行命令

> 更新时间：2026-10-19。
> 状态：入口已重构为 train.py + src/pinn_sr，24 项回归与作者 data/5 的 Mac CPU 20 步通过。用户已反馈旧版本 Linux 500000 步完成；新版本 CUDA 与服务器实际权重尚未独立核验。模块职责、兼容策略见 [src/README](../../src/README.md)。
> 当前可执行入口：[train.py](../../train.py)；测试：[tests/test_train.py](../../tests/test_train.py)。
> [Mac/Linux 环境与复现背景](./01_mac_to_linux_reproduction_plan.md)保留；从头训练现在优先使用本文件，不需要再手改作者 pinn_train.py。

## 1. 最快开始

激活有 torch、numpy 的环境，进入本地仓库：

```bash
conda activate pinn_sr_mac
cd /Users/cyt/projects/i/pinn-code/PINN-Super-resolution
python train.py --help
python train.py --check-only --device cpu
python train.py --device cpu --num-threads 1 --run-tag mac-smoke
```

默认是 data/5、10×100 网络、20 步、128 个物理点，便于首次验证；不是默认长跑 500000 步。若尚未创建 pinn_sr_mac，先按第 5 节配置。命令执行后第一行 RUN_DIR=... 就是本次结果目录。

train.py 不依赖 Mayavi、pandas、SciPy 或 Matplotlib；数值训练只需 torch、numpy。不会读取作者旧完整对象 checkpoint。

## 2. 全部参数

| 参数 | 默认值 | 含义与规则 |
|---|---|---|
| `--data-dir` | `data/5` | 观测 CSV 目录；支持 5、20、100、带空格场景或自定义同格式目录 |
| `--phy-file` | `data/phy.csv` | 三列物理配点文件；没有速度标签 |
| `--model` | `mlp` | 第一版仅实现 mlp，不把其他模型名称当别名 |
| `--hidden-layers` | `10` | 隐藏层数量，不包含输入/输出层 |
| `--hidden-width` | `100` | 每个隐藏层宽度 |
| `--steps` | `20` | Adam 更新的总目标步数；续训时包含已有步数，不是追加次数 |
| `--optimizer` | `adam` | 第一版仅支持 Adam |
| `--lr` | `1e-3` | 真正传给优化器的学习率 |
| `--phy-batch-size` | `128` | 每步物理点数；超出总数时封顶并记录实际值，尾批可能更小 |
| `--phy-sampling` | `sequential` | sequential 为全体配点按顺序循环；random 为每步无放回抽样，各步间可重复 |
| `--weight-data` | `5` | 数据损失权重，非负 |
| `--weight-physics` | `1` | 物理损失权重，非负；两个权重不能同时为零 |
| `--seed` | `1234` | 初始化与采样随机种子；新版导入无作者重置种子的副作用 |
| `--device` | `auto` | auto 选可用 CUDA，否则 CPU；显式 cuda 不可用时失败，不静默转 CPU |
| `--dtype` | `float32` | 可选 float64；float64 是额外数值实验，不是原 float32 默认 |
| `--num-threads` | 未指定 | 可限制 PyTorch CPU 线程；Mac 冒烟建议 1，不据此宣称最佳吞吐 |
| `--log-every` | `1` | 终端及 train.log 的打印间隔；metrics.csv 仍记录每一步 |
| `--save-every` | `10` | checkpoint 间隔；最后一步总会保存一次，不重复写同名文件 |
| `--eval-batch-size` | `1024` | 训练结束后的观测点拟合评价批大小，与训练 batch 不同 |
| `--output-root` | `runs` | 新运行目录的父目录 |
| `--run-tag` | 空 | 可选标签，如 mac-smoke、linux-baseline；不改变模型 |
| `--resume` | 未指定 | 从本入口生成的状态文件续训；始终创建新目录 |
| `--allow-legacy-resume` | 关闭 | 显式允许已核验源码版本的重构前 v1 续训；不是跳过检查，旧权重推理无需此项 |
| `--check-only` | 关闭 | 检查数据、路径、设备和续训兼容性；不训练、不创建输出目录 |

路径规则：相对路径相对于 train.py 所在项目根目录，而不是当前终端工作目录；绝对路径直接使用。因此可以在其他目录执行 `python /绝对路径/train.py ...`。支持 ~ 展开；自定义路径含空格时加引号。

数据要求：

- coordinate.csv 是无表头 N×3 数值数组；
- velocity_x/y/z.csv 分别为 1×N 或 N×1，节点顺序必须与坐标一致；
- phy.csv 是非空 M×3 数值数组；
- 所有值有限；原相对误差的速度零分母直接报错，不偷偷添加 epsilon；
- 每个坐标维度范围必须非零；单一平面的恒定维度需要另定义真实域范围，不能直接作为当前归一化输入；
- 没有企业 DAT 适配器，没有时间 t、MPS、ConFIG 或独立测试集接口。

## 3. 保留原方法与工程改造边界

| 保留 | 新入口的工程处理 |
|---|---|
| Tanh、Xavier、坐标 min-max、uvwp 输出 | 独立 PortableDNN；ub/lb 注册为 buffer，显式 dtype/device |
| 三分量绝对相对误差 sum | 独立迁移作者计算，前向/损失/梯度对照通过；不改为 MSE/mean，不加 epsilon |
| 连续性和三维动量残差 | 独立迁移三维导数与残差；六个可学习 PDE 系数仍零初始化 |
| alpha 注册但没有损失梯度 | 保留，梯度有限检查允许这个参数的 grad=None |
| Adam、默认 data:physics=5:1 | 学习率/权重由参数真正控制 |
| 顺序配点轮询 | 使用清晰游标覆盖尾批，避免原脚本整除边界重复首批的情况 |

权重可配置不代表自动平衡；学习到的 PDE 系数不能无条件当作真实物理参数。数据与物理损失仍是 sum，改变观测数量或 physics batch 会改变损失尺度，实验比较必须明确预算与权重。

## 4. 输出命名、保存与防覆盖

例：

```text
runs/
└── mlp_L10_W100_case5_lr0.001_phy128_wd5wp1_seed1234_mac-smoke__UTC时间__配置hash-唯一ID/
    ├── config.json
    ├── environment.json
    ├── train.log
    ├── metrics.csv
    ├── checkpoints/
    │   ├── step_00000010.pt
    │   └── step_00000020.pt
    └── summary.json
```

- 时间戳使用 UTC，末尾 Z；配置完整值以 config.json 为准，文件夹名只是关键参数摘要。
- 每次运行都是新目录，同参数重复执行也不会覆盖；目录、JSON、日志和 checkpoint 都采用独占创建。
- 拒绝写入 data/、pinn/、src/、docs/、tests/、.git/ 和原 result_model_pinn* 目录，校验经过 symlink 解析的真实路径。
- 状态文件包含模型参数、归一化 buffer、优化器状态、步数、配置、采样状态和 Torch RNG；使用 weights_only=True 恢复。
- 不把新状态文件交给作者原 pinn_predict_plane.py，它要求完整 PINN 对象；新状态需用本入口 restore_checkpoint。
- 最后 summary.json 记录 completed/failed/interrupted、checkpoint 检查及 train_fit；失败不会覆盖其他运行。
- /runs/ 已加入新 .gitignore；不删除或取消追踪作者已有缓存/权重。

train_fit 是同一训练观测点上的 MAE、RMSE、Relative L2，不是独立测试或三维超分辨率结论。日志 loss 是该步更新前的损失；最终 train_fit 使用完成更新后的模型。CUDA 峰值包括完整监督数据和模型，null 表示 CPU 不适用，不是显存为 0。

## 5. Mac 环境与测试指令

### 5.1 新环境（安装组合待用户实际验证）

```bash
conda create -n pinn_sr_mac python=3.11 -y
conda activate pinn_sr_mac
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 numpy==1.26.4
python -m pip check
cd /Users/cyt/projects/i/pinn-code/PINN-Super-resolution
```

版本依据 [PyTorch 官方历史安装页](https://pytorch.org/get-started/previous-versions/)。实际开发测试使用现有 Conda base：Python 3.13.5 / torch 2.9.0 / numpy 2.3.4；没有创建或更改你的环境，不能把建议 Python 3.11 组合写成已验证。

### 5.2 自动测试、只检查、20 步冒烟

```bash
python -m unittest discover -s tests -v

python train.py --check-only \
  --data-dir data/5 --phy-file data/phy.csv --device cpu

python train.py \
  --data-dir data/5 --phy-file data/phy.csv \
  --model mlp --hidden-layers 10 --hidden-width 100 \
  --steps 20 --optimizer adam --lr 1e-3 \
  --phy-batch-size 128 --phy-sampling sequential \
  --weight-data 5 --weight-physics 1 \
  --device cpu --dtype float32 --num-threads 1 --seed 1234 \
  --log-every 1 --save-every 10 --eval-batch-size 1024 \
  --output-root runs --run-tag mac-smoke
```

验收：20 步 loss/gradient/parameters 有限；新目录有 10、20 步状态文件；终端有 checkpoint round-trip PASS；summary.json 的 status=completed。不要求 loss 单调或达到论文精度。

### 5.3 Mac 完整训练命令（本地工程预算）

```bash
python train.py \
  --data-dir data/100 --phy-file data/phy.csv \
  --model mlp --hidden-layers 10 --hidden-width 100 \
  --steps 10000 --optimizer adam --lr 1e-3 \
  --phy-batch-size 256 --phy-sampling sequential \
  --weight-data 5 --weight-physics 1 \
  --device cpu --dtype float32 --num-threads 1 --seed 1234 \
  --log-every 50 --save-every 1000 --eval-batch-size 1024 \
  --output-root runs --run-tag mac-train-100
```

这是完整执行链路和全部核心参数示例，但使用 10000 步、256 physics batch 的本地预算，**不等于作者 500000 步 / 31000 batch 的实验**。未执行该长训练，收敛与耗时不能由 20 步推断。若只想学习流程，可先把 steps 改为 1000；正式论文预算优先放 Linux。

Mac 不支持 CUDA；第一版不启用 MPS，不要加 --device mps。

## 6. Linux 环境、测试与完整训练

### 6.1 服务器检查与环境配置（全部未实测）

```bash
nvidia-smi
conda create -n pinn_sr_linux python=3.11 -y
conda activate pinn_sr_linux
python -m pip install --upgrade pip

# 仅在 GPU 架构和驱动兼容 CUDA 12.6 wheel 时使用这一示例
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==1.26.4
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CUDA wheel 根据实际服务器选择，不能把 cu126 无条件用于所有服务器。[官方版本表](https://pytorch.org/get-started/previous-versions/)；nvidia-smi 的 CUDA Version 不等于 PyTorch runtime。服务器需同步包含 train.py 的本地代码版本，本轮未自动 commit/push。

进入服务器的 PINN-Super-resolution 项目根目录后执行下面命令。所有 Linux GPU 命令显式指定 cuda，不使用会回退 CPU 的 auto。

### 6.2 自动测试与 GPU 冒烟

```bash
python -m unittest discover -s tests -v
python train.py --check-only --data-dir data/5 --phy-file data/phy.csv --device cuda

python train.py \
  --data-dir data/5 --phy-file data/phy.csv \
  --model mlp --hidden-layers 10 --hidden-width 100 \
  --steps 20 --optimizer adam --lr 1e-3 \
  --phy-batch-size 128 --phy-sampling sequential \
  --weight-data 5 --weight-physics 1 \
  --device cuda --dtype float32 --seed 1234 \
  --log-every 1 --save-every 10 --eval-batch-size 1024 \
  --output-root runs --run-tag linux-smoke
```

单元测试主要检查 CPU 逻辑；通过 unittest 不代表 CUDA 已验证。GPU 冒烟还需确认 environment.json 的 device=cuda、GPU 型号，以及 summary.json 的 CUDA 峰值和 roundtrip。

### 6.3 Linux 完整基线命令

```bash
python train.py \
  --data-dir data/100 --phy-file data/phy.csv \
  --model mlp --hidden-layers 10 --hidden-width 100 \
  --steps 500000 --optimizer adam --lr 1e-3 \
  --phy-batch-size 31000 --phy-sampling sequential \
  --weight-data 5 --weight-physics 1 \
  --device cuda --dtype float32 --seed 1234 \
  --log-every 50 --save-every 1000 --eval-batch-size 1024 \
  --output-root runs --run-tag linux-baseline-100
```

31000 个物理点含二阶自动微分，可能 OOM；先测较小 batch 的 20 步，再逐步放大。可用 4096/8192 等单独预算实验，但必须记录这是不同 physics batch，不能混为严格相同协议。建议在 tmux 或平台任务调度系统中长跑，不直接占用未获分配的共享 GPU。

仅保持这些超参数不等于复现论文指标：仍需核对原文优化器、采样、参考集合和评价区域。

## 7. 续训与读取新状态文件

### 7.1 安全续训

从终端 RUN_DIR 找到需要的 checkpoint；下例的尖括号路径必须换成真实路径：

```bash
python train.py \
  --data-dir data/5 --phy-file data/phy.csv \
  --model mlp --hidden-layers 10 --hidden-width 100 \
  --steps 40 --optimizer adam --lr 1e-3 \
  --phy-batch-size 128 --phy-sampling sequential \
  --weight-data 5 --weight-physics 1 \
  --device cpu --dtype float32 --num-threads 1 --seed 1234 \
  --log-every 1 --save-every 10 \
  --resume "runs/<上次运行目录>/checkpoints/step_00000020.pt" \
  --output-root runs --run-tag mac-resume
```

steps=40 表示总共做到第 40 步，从第 20 步文件继续只增加 20 步。新目录保存 30、40 步状态，不重写旧目录。需重复原网络、学习率、损失、数据、seed、采样参数；改变这些参数或数据会被拒绝。新版普通续训要求维护源码哈希与方法契约一致；仅已核验的重构前源码组合可以通过 --allow-legacy-resume 显式迁移，未知源码不能绕过检查。

允许更改总步数、日志/保存频率、输出目录和设备；跨 CPU/CUDA 迁移不承诺逐位一致，CUDA 结果也未实测。只接受新格式状态文件，不通过 weights_only=False 打开作者旧模型。

旧版 v1 迁移：在上面续训命令中追加 --allow-legacy-resume，且 --resume 指向旧文件。其余参数必须与旧实验一致；无需为了推理/绘图再次训练。该选项只允许 checkpoint.py 中预先核验的源码哈希组合，服务器实际文件尚未核验，若提示未知来源需先比对源码。新运行的 config.json 会记录 resume_compatibility。

### 7.2 数值读取示例

在项目根目录运行，先换成真实 checkpoint 路径：

```python
from pathlib import Path
import torch
from train import restore_checkpoint

model, state = restore_checkpoint(
    Path("runs/<运行目录>/checkpoints/step_00000020.pt"), device="cpu"
)
model.net.eval()
# 示例坐标，不代表真实参考观测或预测准确性验证
xyz = torch.tensor([[1.0, 0.8, 0.7]], dtype=model.net.lb.dtype)
with torch.no_grad():
    uvwp = model.net(xyz)
print("trained step:", state["step"], "prediction:", uvwp)
```

此代码只展示新格式加载方式。它不画图、不验证任意坐标的真实误差。训练拟合指标已由 summary.json 提供。

## 8. 当前真实验证与边界

| 项目 | 2026-10-14 状态 |
|---|---|
| 10 项 CPU 回归测试 | 已通过 |
| 默认数学形式与作者代码前向/loss/梯度对照 | 已通过；合成小网络测试 |
| 指定 CPU、学习率、权重生效 | 已通过；包括模拟 CUDA 可见而显式选 CPU |
| 原目录/符号链接防写入、同参数新目录、同名 checkpoint 拒写 | 已通过 |
| CPU 短续训与不中断训练结果一致 | 已通过；合成小网络、同 CPU 配置 |
| 作者 data/5，10×100 网络，20 步，physics batch=128 | 已通过；334 个观测点、213624 个总配点 |
| 新 checkpoint 保存/恢复预测一致 | 已通过 |
| data/100 长训练、论文独立精度 | 未运行 |
| Python 3.11 隔离环境、Linux CUDA、MPS、Mayavi | 未验证 |

20 步只是工程测试，最终训练点 RMSE 仍较大，不能标为论文精度已复现。运行输出在本机 /tmp/pinn-sr-training-validation-20261014 的独立实验目录；临时目录可能清理，长期实验请使用持久存储。没有修改作者原 .py、CSV 或预训练权重，没有训练企业数据。

第一步推荐：环境就绪后执行第 5.2 节；Mac 通过，再复制同版本代码到 Linux 执行第 6.2 节，最后才做第 6.3 节长训练。
