---
title: PINN-Super-resolution：Mac 到 Linux 复现计划
summary: 先在 Apple Silicon Mac 上验证环境、数据、预训练推理和小规模二阶自动微分训练，再在 Linux CUDA 服务器上重复检查并开展正式实验。区分工程跑通、论文指标复现和企业双截面数据适配。
type: research-plan
status: to-run
created: 2026-10-14
updated: 2026-10-14
papers:
  - "An airflow velocity field reconstruction method with sparse or incomplete data using physics-informed neural network; DOI: 10.1016/j.jobe.2024.109231"
github_repositories:
  - https://github.com/yigeyouzia/PINN-Super-resolution
  - https://github.com/wanghuan10/PINN-Super-resolution
data:
  - 作者公开 data/5、data/20、data/100、局部缺失数据与 data/phy.csv
  - 企业双截面 DAT；仅用于后续本地适配，不上传或随代码公开
---

# Mac → Linux 复现计划

> 整理日期：2026-10-14。
> 当前状态：已读取源码、检查当前 Mac 的基础环境；本计划中的独立环境安装、模型加载、短训练、Linux 实验均尚未执行。
> 路线：Mac CPU 跑通 → Linux CUDA 跑通 → 正式基线 → 稀疏/缺失实验 → 企业数据适配。
> 根目录 [README](../../README.md) 负责方法与代码导读；[旧 Linux 指南](./00_linux_environment_and_baseline.md) 保留历史记录，本文件作为当前跨平台执行入口。

## 1. 先明确“复现”的三个层次

| 层次 | 目标 | 完成证据 |
|---|---|---|
| 工程跑通 | 环境、数据读取、前向预测、PDE 二阶导数、反向更新和保存/恢复可用 | 命令、日志、有限数检查、checkpoint 往返检查 |
| 论文复现 | 按论文及代码协议训练，核对参考结果 | 训练预算、评价区域、原文表格、误差与资源结果；不能只看 loss |
| 企业迁移 | 用企业观测和正确物理约束重建/预测 | 几何、方程、数据权限、时间或工况对应关系、独立评价真值 |

本项目当前为 xyz → uvwp 的稳态三维坐标 PINN，不是历史窗口预测器。不在首次跑通时同时添加时间、Transformer、ConFIG 或新损失。

## 2. 当前 Mac 与环境选择

2026-10-14 在当前工具执行环境实际检查到：

| 项目 | 结果 |
|---|---|
| 操作系统 / 架构 | macOS 15.3 / arm64 |
| Conda | /Users/cyt/miniconda3 |
| Conda base Python | 3.13.5 |
| base 依赖 | torch 2.9.0；numpy 2.3.4；pandas 2.3.3；scipy 1.16.3；matplotlib 3.10.9 |
| CUDA | False；Mac 不使用 NVIDIA CUDA |
| MPS | is_built=True；本次受限执行环境中 is_available=False |
| Mayavi | 未安装 |

这里的 MPS 检测不等于判定机器硬件不支持 MPS，应在自己的终端复查。即使可用，原 network.py、pinn.py、pinn_train.py 也只选择 CUDA 或 CPU，并不会自动启用 MPS。

**首选 CPU 冒烟测试**，避免同时修改多个文件的设备逻辑。MPS 作为可选后续实验：需要统一设备、正确搬运归一化上下界，并验证二阶空间导数及其反向传播；不能只凭张量创建成功就判断 PINN 训练兼容。[MPS 官方说明](https://docs.pytorch.org/docs/main/notes/mps.html)

不在 base 中升级/降级依赖；新建隔离环境。下面版本是建议的起始组合，尚未安装验证，不是原作者锁定环境：

- Python 3.11
- torch 2.9.0
- numpy 1.26.4、pandas 2.2.3、scipy 1.14.1、matplotlib 3.9.4

### 2.1 Mac 安装命令（待执行）

```bash
conda create -n pinn_sr_mac python=3.11 -y
conda activate pinn_sr_mac
python -m pip install --upgrade pip
python -m pip install torch==2.9.0
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 matplotlib==3.9.4
python -m pip check

export PINN_SR_ROOT="/Users/cyt/projects/i/pinn-code/PINN-Super-resolution"
cd "$PINN_SR_ROOT"
export PINN_SR_RUNS="$(mktemp -d "${TMPDIR:-/tmp}/pinn-sr-runs.XXXXXX")"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PINN_SR_RUNS/matplotlib-cache"
python -m pip freeze > "$PINN_SR_RUNS/requirements-mac.txt"
git rev-parse HEAD
```

每次实验创建新的 PINN_SR_RUNS，输出先放该临时目录；完成后保留到个人实验存储，不覆盖仓库预训练权重。临时目录可能被系统清理，不适合长期归档。

torch 的 macOS 安装不带 CUDA index；不必安装 torchvision/torchaudio。[官方安装页](https://pytorch.org/get-started/locally/) / [2.9.0 历史安装命令](https://pytorch.org/get-started/previous-versions/)

### 2.2 环境验收（Mac / Linux 共用）

```bash
python - <<'PY'
import sys, platform
import torch, numpy, pandas, scipy, matplotlib
print("python:", sys.version)
print("platform:", platform.platform(), platform.machine())
for module in (torch, numpy, pandas, scipy, matplotlib):
    print(module.__name__, module.__version__)
print("CUDA:", torch.cuda.is_available(), "runtime:", torch.version.cuda)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU tensor:", torch.rand(4, device="cuda").sum().item())
PY
```

Mac 的 CUDA=False 正常。Linux 若目标为 NVIDIA GPU，则 CUDA=True 且 GPU 张量运算成功才算 GPU 环境通过。记录完整输出。

## 3. M00：数据读取与数值检查（待执行）

在前述激活环境且设置路径后：

```bash
cd "$PINN_SR_ROOT/pinn"
python - <<'PY'
import numpy as np
from pinn_train import load_data, load_phy
for case in ("5", "20", "100"):
    x, y, z, u, v, w, ub, lb, *_ = load_data(f"../data/{case}/")
    xyz = np.concatenate((x, y, z), axis=1)
    velocity = np.concatenate((u, v, w), axis=1)
    assert xyz.shape == velocity.shape and xyz.shape[1] == 3
    assert np.isfinite(xyz).all() and np.isfinite(velocity).all()
    assert (ub > lb).all()
    print(case, "shape:", xyz.shape,
          "zero components:", (velocity == 0).sum(axis=0),
          "min |velocity|:", np.abs(velocity).min(axis=0))
phy = np.concatenate(load_phy("../data/"), axis=1)
assert phy.shape[1] == 3 and np.isfinite(phy).all()
print("phy:", phy.shape, "min/max:", phy.min(axis=0), phy.max(axis=0))
PY
```

验收：5/20/100 观测点数应分别为 334/2875/8297；物理点总数以本地实际输出为准，不能把 31000 的物理 batch 当总数。记录零值和近零值，因为原相对误差损失直接除以真实速度。

读取成功不代表几何、单位、评价协议已验证。

## 4. M01：预训练数值推理（待执行，不依赖 Mayavi）

**安全前置条件**：只有明确信任作者 checkpoint 及下载来源才执行本节。旧文件保存完整 Python 对象，weights_only=False 会执行 pickle 反序列化；来源不可信则跳过 M01，直接用随机初始化做 M02。不以降级 PyTorch 或全局关闭安全检查绕过不可信文件。

```bash
cd "$PINN_SR_ROOT/pinn"
python - <<'PY'
import numpy as np
import torch
from pinn import PINN
from pinn_train import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../result_model_pinn_100/model_pinn_197000",
                   map_location=device, weights_only=False)
model.net.eval()
x, y, z, u, v, w, *_ = load_data("../data/100/")
xyz = np.concatenate((x, y, z), axis=1)
target = np.concatenate((u, v, w), axis=1)
with torch.no_grad():
    inputs = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    out = model.net(inputs)
    pred = out[:, :3].cpu().numpy()
assert out.shape == (8297, 4)
assert np.isfinite(pred).all()
print("device:", device, "input:", xyz.shape, "output:", tuple(out.shape))
print("MAE uvw:", np.abs(pred-target).mean(axis=0))
print("RMSE uvw:", np.sqrt(((pred-target)**2).mean(axis=0)))
print("prediction finite:", np.isfinite(pred).all())
PY
```

本节用 model.net 直接前向，不调用带自动微分逻辑的 predict。map_location 也会映射旧对象中的张量；DNN.ub/lb 不是注册 buffer，不应只做 model.net.to(device) 就假定所有上下界随模型移动。

**验收**：模型可加载、输入 (8297,3)、输出 (8297,4)、误差有限。此处是已有观测点上的拟合诊断，不能写成独立测试精度、论文表格复现或三维体积泛化。

## 5. M02：20 步短训练与保存/恢复（待执行）

下面直接调用原 PINN 类，不改原始 .py 文件；保持原 10×100 网络、原损失和 Adam，仅缩小训练规模与预算。使用 data/5 的全部 334 个观测和固定抽样的 128 个物理点，测试二阶空间导数、参数反向更新和 checkpoint 往返。

**不是正式训练协议**：配点固定、batch 缩小、20 步均只为流程验证，不能与原完整实验比较。必须在 M00 完成后运行；若原损失因零值/近零值失效，记录失败，不自动改损失后仍标为“原实现复现”。

```bash
cd "$PINN_SR_ROOT/pinn"
python - <<'PY'
import os, time
from pathlib import Path
import numpy as np
import torch
from pinn import PINN
from pinn_train import load_data, load_phy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.environ.get("PINN_SR_REQUIRE_CUDA") == "1":
    assert device.type == "cuda", "Linux GPU 测试不允许静默退回 CPU"
torch.manual_seed(1234)
rng = np.random.default_rng(1234)
x, y, z, u, v, w, ub, lb, ub_v, lb_v = load_data("../data/5/")
xyz = np.concatenate((x, y, z), axis=1)
vel = np.concatenate((u, v, w), axis=1)
assert np.isfinite(vel).all() and not np.any(vel == 0), "原损失存在零分母"
phy_all = np.concatenate(load_phy("../data/"), axis=1)
phy_index = rng.choice(len(phy_all), size=min(128, len(phy_all)), replace=False)
a = torch.as_tensor(xyz, dtype=torch.float32, device=device)
b = torch.as_tensor(vel, dtype=torch.float32, device=device)
c = torch.as_tensor(phy_all[phy_index], dtype=torch.float32, device=device)
layers = [3] + [100] * 10 + [4]
model = PINN(layers, ub, lb, ub_v, lb_v)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
start = time.perf_counter()
for step in range(20):
    _, total, data, physics = model.closure(a, b, c)
    assert all(torch.isfinite(t).all() for t in (total, data, physics)), "损失非有限"
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.net.parameters()), "梯度非有限"
    model.adam.step()
    assert all(torch.isfinite(p).all() for p in model.net.parameters()), "参数非有限"
    print(step + 1, float(total.detach()), float(data.detach()),
          float(physics.detach()), flush=True)
if device.type == "cuda":
    torch.cuda.synchronize()
print("device:", device, "wall_sec:", time.perf_counter()-start)
if device.type == "cuda":
    print("peak_allocated_MiB:", torch.cuda.max_memory_allocated()/1024**2)
    print("peak_reserved_MiB:", torch.cuda.max_memory_reserved()/1024**2)

run_dir = Path(os.environ["PINN_SR_RUNS"])
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint = run_dir / "smoke_state_dict.pt"
torch.save({
    "state_dict": model.net.state_dict(),
    "optimizer": model.adam.state_dict(),
    "layers": layers, "ub": ub.tolist(), "lb": lb.tolist(),
    "ub_velocity": ub_v.tolist(), "lb_velocity": lb_v.tolist(),
    "step": 20, "seed": 1234, "case": "5",
    "phy_index": phy_index.tolist()
}, checkpoint)
state = torch.load(checkpoint, map_location=device, weights_only=True)
restored = PINN(state["layers"], np.array(state["ub"]), np.array(state["lb"]),
                np.array(state["ub_velocity"]), np.array(state["lb_velocity"]))
restored.net.load_state_dict(state["state_dict"])
restored.adam.load_state_dict(state["optimizer"])
with torch.no_grad():
    before = model.net(a)
    after = restored.net(a)
torch.testing.assert_close(before, after)
assert torch.isfinite(after).all()
print("checkpoint round-trip PASS:", checkpoint)
PY
```

验收：20 次更新、loss/gradient/parameters 有限、保存恢复预测一致。**不要求 20 步 loss 严格单调或达到论文精度**。本检查验证保存/恢复与优化器状态可加载，不等于已验证中断后逐步完全一致的训练恢复。

若 Mac CPU 太慢，可单独把物理点 128 改为 32，记录修改；不要把不同 batch 的 sum loss 直接比较。正式网络不缩小，以便避免把结构变化混入“环境跑通”。

## 6. Linux CUDA 环境与迁移（待执行）

先取得 GPU 型号、显存、驱动、发行版；当前尚无这些信息，不能确定最终 CUDA wheel。

```bash
uname -a
nvidia-smi
conda --version
conda create -n pinn_sr_linux python=3.11 -y
conda activate pinn_sr_linux
python -m pip install --upgrade pip
```

例如，**仅在驱动、GPU 架构与 CUDA 12.6 wheel 兼容时**：

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 matplotlib==3.9.4
python -m pip check
```

这是条件式示例，不是已确认适用的服务器配置。[PyTorch 官方历史版本与 CUDA wheel](https://pytorch.org/get-started/previous-versions/) 提供其他组合；若 GPU 架构、驱动不支持，需要重新选版本，而不是修改共享服务器驱动。nvidia-smi 的 CUDA Version 不等于 torch.version.cuda 或本机 nvcc 版本。

服务器无项目时克隆个人 fork；已有时检查工作区后再同步，不能覆盖未提交修改。**本次文档尚未推送**，服务器克隆不一定含本文件，需明确同步同一代码版本和文档。

```bash
git clone https://github.com/yigeyouzia/PINN-Super-resolution.git
cd PINN-Super-resolution
export PINN_SR_ROOT="$PWD"
export PINN_SR_RUNS="$(mktemp -d /tmp/pinn-sr-runs.XXXXXX)"
export PINN_SR_REQUIRE_CUDA=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PINN_SR_RUNS/matplotlib-cache"
python -m pip freeze > "$PINN_SR_RUNS/requirements-linux.txt"
git rev-parse HEAD
```

随后依次重复第 2.2 节、M00、M01（仅可信 checkpoint）、M02。Linux 显式要求 CUDA=True；M02 启用 CUDA 失败保护。

Mac/Linux 使用相同数据、模型和种子，误差可作一致性诊断，但浮点结果不要求逐位一致。不要把 Mac 的整个 Conda 目录或 freeze 文件原样搬到 Linux 安装；wheel 与底层平台不同，分别保存环境记录。

Linux 无头绘图优先 Matplotlib Agg 保存 PNG；不在首次验证中安装 Mayavi、VTK、Qt 或运行原 GUI 预测入口。正式实验使用个人持久化运行目录，勿长期只放 /tmp。

## 7. 后续正式复现与协议核对

| 阶段 | 内容 | 最低交付 |
|---|---|---|
| L00 | Linux 重复环境、数据、推理与短训练 | CUDA 真正可用；日志、版本、峰值显存 |
| L01 | 原代码基线，从头训练 data/100 | 完整配置、种子、步数、数据/physics loss、checkpoint、时间 |
| L02 | 5/20/100 稀疏观测对比 | 固定评价区域与参考数据，分别报告拟合与留出点误差 |
| L03 | 局部缺失区域实验 | 缺失区与非缺失区误差分别报告；核对左右与 upper/lower 的映射 |
| L04 | ConFIG 等训练方法 | 固定模型、损失、采样、训练预算与硬件，单独测梯度与成本 |
| L05 | 企业双截面适配 | DAT 数据卡、同步关系、正确方程、独立截面/体积参考结果 |
| L06 | 时序场预测 | 确认真实时间与间隔后，非稳态残差与未来留出协议 |

正式训练不要原样一上来跑 500000 步。先单独增加可配置入口、输出隔离与定量评价，再按目标精度、预算和论文协议决定训练步数。仅本轮文档修改，尚未新增该训练入口。

旧脚本保存到不存在的 result_model_pinn/，而且不区分实验；后续实施时创建专属输出目录。绝不覆盖随仓库下载的预训练权重。

### 7.1 数值/方法改动必须分轨

- 原样基线：原相对误差 sum、5:1 权重、可学习 PDE 系数；记录失败也属于有效复现证据。
- 工程修复版：路径、日志、可配置 budget、稳定保存；记录代码版本。
- 方法/稳定性变体：分母 epsilon、归一化 MAE/MSE、sum → mean、固定物理系数或新优化器；不能静默修改后仍声称完全复现原实现。

不同物理 batch 下 sum loss 的量级会改变，必须记录 actual batch，不能仅凭 loss 更低认定方法更好。拟合误差、物理残差、推理耗时、训练耗时、峰值显存分别报告。

### 7.2 论文指标复现不能由短测试代替

需核对论文测量预处理、5/20/100 的实际含义、训练与参考点关系、评价区域、网络/优化器和代码差异。论文数据百分比与体平均后的点数比例不是一回事。

data/100 可作为公共参考集合，但可能与训练点重叠；要标清重叠与非重叠误差。建立自己留出协议属于新基准，不自动等于作者论文原测试协议。压力无标签时，不宣称压力准确性已验证。

## 8. 企业数据迁移边界

当前企业文件是两个固定垂直面，每组只有 xyz、uvw；两面各 45 组文件且 SOLUTIONTIME=0。不能仅凭编号当连续时间，不能将不同状态拼成只有 xyz 输入的单一稳态数据集。

适配前确认：

- [ ] 同编号 X/Y 是否同一时刻、同一工况；
- [ ] 45 组代表时间、工况还是数值迭代；
- [ ] 几何、流体区域、边界、单位与物理参数；
- [ ] CFD 使用瞬态 NS、RANS/LES 还是其他模型；不能无条件套稳态层流残差；
- [ ] 独立截面或体积参考数据，以及企业数据使用/公开权限。

单独使用一个截面时，某坐标恒定会使原 min-max 的 ub-lb=0。后续应使用真实域范围或显式处理恒定维度，不能直接沿用单面极值归一化。

## 9. 进度与实验记录

| 项目 | 当前状态 |
|---|---|
| README / 源码静态核对 | 已完成 |
| 当前 Mac base 依赖/设备查询 | 已完成；不是项目运行测试 |
| 隔离环境安装与 pip check | 待执行 |
| M00 数据检查命令 | 待执行 |
| M01 预训练推理 | 待执行；需信任 checkpoint |
| M02 CPU 短训练/保存恢复 | 待执行 |
| Linux CUDA 与正式实验 | 待执行 |

每次实际运行后填写，不用“预计成功”代替结果：

```text
日期 / 机器 / 操作系统：
代码 commit / 未提交 diff：
环境名称 / Python / torch / CUDA runtime / driver：
数据组 / 数据 hash / seed / 物理点抽样：
输入输出形状 / 训练步数 / 实际 batch：
MAE / RMSE uvw（说明评价点是否见过）：
loss_data / loss_phy / 是否有限：
wall time / peak allocated / peak reserved：
checkpoint 路径 / 保存恢复测试：
状态：未运行 / 失败 / 部分跑通 / 工程跑通 / 指标复现
失败原因 / 修复措施 / 与原实现差异：
日志与环境快照位置：
```

## 10. Markdown 公式约定

独立数学公式使用单独成行的 `$$` 包围，前后留空行；行内数学使用 `$...$`。不使用两个反引号包住 `math`，也不把要渲染的公式放入普通代码块。源代码变量名仍可使用行内代码。

VS Code 使用 Markdown: Open Preview 并确认 markdown.math.enabled=true；编辑器源文件本身显示 LaTeX 是正常现象，要在预览中检查。GitHub 使用其 Markdown 数学渲染。[GitHub 数学语法](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)

当前下一步：先执行第 2 节 Mac 隔离环境配置，再依次 M00 → 可信权重 M01 → M02；没有训练结果前不标为已复现。

