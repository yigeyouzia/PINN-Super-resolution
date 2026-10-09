# 实验 00｜Linux 环境搭建与 PINN-Super-resolution 基线验证

> **仓库**：https://github.com/yigeyouzia/PINN-Super-resolution  
> **上游**：https://github.com/wanghuan10/PINN-Super-resolution  
> **实验状态**：`待执行（实验说明与运行命令已准备，尚无 Linux/GPU 实测结果）`  
> **创建日期**：2026-10-09  
> **目标**：先建立稳定的服务器环境、验证预训练 checkpoint，然后再做短训练和 5/20/100 三组正式实验。  
> **代码结构说明**：参见仓库根目录 [README.md](../../README.md)。

## 0. 为什么先做这个实验

当前代码是 **3D 稳态 Navier–Stokes PINN**，输入 `(x,y,z)`，输出 `(u,v,w,p)`，由数据监督损失与连续性/动量方程的物理残差联合优化；使用十层、每层 100 个单元的 Tanh MLP。

仓库已经提供 `result_model_pinn_100/model_pinn_197000`，因此**第一项实验不是直接运行 500,000 步训练，而是验证该预训练模型可以被当前 Linux/PyTorch 环境正确加载和预测**。这比盲目重训更容易定位依赖、序列化、GPU 和数据路径问题。

**本文与实验结果的边界**：文档中提到的 8,297 个观测坐标来自代码及数据文件检查；模型加载、GPU、评估数值均**没有在用户服务器上执行**，请在「执行记录」中补齐。

## 1. 服务器准备与环境记录（第 0 步）

登录 Linux 服务器：

```bash
nvidia-smi
python --version
conda --version
git --version
```

在实验记录中保存以下信息：

| 项目 | 实际值（运行后填写） |
| --- | --- |
| 日期 / 服务器 | 待填写 |
| Linux 发行版 / 内核 | 待填写 |
| GPU 型号 / 数量 | 待填写 |
| 显存容量 / 空闲显存 | 待填写 |
| NVIDIA Driver Version | 待填写 |
| `nvidia-smi` 显示的 CUDA Version | 待填写 |
| Conda 版本 | 待填写 |
| Python 版本 | 待填写 |
| PyTorch 版本 / `torch.version.cuda` | 待填写 |
| Git commit | 待填写 |
| 实验运行结果 | 待填写 |

> `nvidia-smi` 显示的 CUDA Version 是**驱动支持的 CUDA 版本信息**，不能直接认为是本机 `nvcc` 版本或 PyTorch wheel 内 CUDA runtime 版本。若 `nvidia-smi` 失败，先解决驱动/容器可见性问题。

## 2. 创建专用 Python 环境（第 1 步）

避免和其他 PINN/ConFIG/PINNsFormer 环境混用。以下采用 **Conda + Python 3.10**：

```bash
conda create -n pinn_sr python=3.10 -y
conda activate pinn_sr
python -m pip install --upgrade pip
```

### 2.1 安装 PyTorch（先根据 GPU 与驱动选 wheel）

此仓库**没有** `requirements.txt` / `environment.yml` 和原作者锁定的版本。为了使首次验证更可控，可以先尝试 PyTorch 2.7.1 的指定 CUDA wheel；**不能仅根据命令推断该 wheel 适合你的服务器**。

示例：**仅在 NVIDIA 驱动/GPU 兼容 CUDA 12.6 wheel 时**执行：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

其他情况应从 [PyTorch 官方历史版本安装页](https://pytorch.org/get-started/previous-versions/) 选择兼容的 CUDA wheel（例如对应版本提供的 `cu118`、`cu128`）或 CPU wheel；**不要将示例 `cu126` 无条件套用到所有服务器**。

安装训练数据处理和绘图的基础依赖：

```bash
python -m pip install "numpy<2" pandas scipy matplotlib
```

说明：

- **首次预训练验证不需要安装 Mayavi**：原 `pinn_predict_plane.py` 在模块导入阶段就需要 `mayavi` / VTK / GUI，在无桌面的 Linux 服务器上容易失败；先使用下面的纯数值推理。
- `torch`、`numpy`、`pandas` 是推理必需；`scipy`、`matplotlib` 供后续可视化使用。
- 这些是**建议的起始环境**，不是作者发布的已验证依赖清单。跑通后应另行导出完整环境版本记录。

### 2.2 检查 PyTorch GPU 可用性

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(100, 100, device="cuda")
    print("GPU tensor test:", x.mean().item())
PY
```

**通过条件**：若计划 GPU 复现，`CUDA available: True` 且 GPU tensor test 不报错。若 CUDA 不可用，可以先进行 CPU 小规模诊断，但不要把 CPU 可运行视作 GPU 环境已配置成功。

## 3. 获取个人 Fork（第 2 步）

如果服务器**尚未克隆**：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/yigeyouzia/PINN-Super-resolution.git
cd PINN-Super-resolution
```

如果已经克隆过，请进入原工作目录，在确认**没有未提交改动**后用 `git pull --ff-only` 更新；不必再次 clone。

核查代码、数据、权重，并记录实际提交：

```bash
git rev-parse HEAD
ls pinn/
ls data/100/
ls -lh result_model_pinn_100/model_pinn_197000
```

关键文件：

```text
pinn/network.py                       # MLP
pinn/pinn.py                          # PINN、PDE 自动微分、loss
pinn/pinn_train.py                    # 数据加载与训练入口
pinn/pinn_predict_plane.py            # 预测和 Mayavi 可视化入口
data/100/coordinate.csv               # 8297 x 3
data/100/velocity_{x,y,z}.csv         # 各 1 x 8297（格式应运行校验）
data/phy.csv                          # 物理配点
result_model_pinn_100/model_pinn_197000  # 现成 checkpoint
```

注意：原工程多个路径都写成 `../data/...`，**命令必须在 `pinn/` 目录执行**，不能简单地从仓库根目录运行入口脚本。

## 4. 实验 E00：加载预训练模型并验证预测（第 3 步，优先）

### 4.1 目的

- 验证完整对象 checkpoint 在当前 PyTorch/Python 环境可加载；
- 使用 `data/100/` 对应坐标，计算 `(u,v,w)` 预测；
- 打印输入/输出形状、每个速度分量的 MAE、RMSE、预测是否有限；
- **不修改原始代码，不依赖 Mayavi，不进行长训练**。

### 4.2 执行命令

在仓库根目录下执行：

```bash
cd pinn
python - <<'PY'
import numpy as np
import torch

from pinn import PINN
from pinn_train import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../result_model_pinn_100/model_pinn_197000"

# 原始模型是 torch.save(pinn, ...) 保存的完整 Python 对象。
# 只能对可信来源 checkpoint 使用 weights_only=False。
model = torch.load(model_path, map_location=device, weights_only=False)
model.net.eval()

x, y, z, vx, vy, vz, *_ = load_data("../data/100/")
xyz = np.concatenate([x, y, z], axis=1)
target = np.concatenate([vx, vy, vz], axis=1)

inputs = torch.as_tensor(xyz, dtype=torch.float32, device=device)
with torch.no_grad():
    outputs = model.net(inputs)
    predicted = outputs[:, :3].cpu().numpy()

mae = np.mean(np.abs(predicted - target), axis=0)
rmse = np.sqrt(np.mean((predicted - target) ** 2, axis=0))

print("\n===== E00 Pretrained Model Test =====")
print("Device:", device)
print("Input shape:", xyz.shape)
print("Output shape:", tuple(outputs.shape))
print("MAE  (u,v,w):", mae)
print("RMSE (u,v,w):", rmse)
print("Prediction finite:", bool(np.isfinite(predicted).all()))
print("====================================")
PY
```

**序列化风险**：`weights_only=False` 会允许 Python pickle 反序列化逻辑，**绝不可用于不可信 checkpoint**。若加载不成功，先保存异常全文，检查版本/模块路径，不建议通过关闭安全检查来处理来源不明的文件。

### 4.3 验收标准

- [ ] 成功从指定路径加载 `model_pinn_197000`
- [ ] `Input shape == (8297, 3)`
- [ ] `Output shape == (8297, 4)`
- [ ] `Prediction finite: True`
- [ ] MAE 与 RMSE 输出为有限值
- [ ] 记录运行设备、Python/PyTorch/CUDA 版本、Git commit

**重要解释**：这里的真实值来自 `data/100/` 的观测点，可能包含训练监督点，所以这是**观测点拟合诊断**，不是独立测试集上的超分辨率泛化评估，也不能拿来与论文测试误差直接比较。

### 4.4 E00 执行记录（运行后填写）

```text
执行日期：
机器/GPU：
分支/Commit：
Conda env：
Python：
PyTorch：
CUDA available：
模型路径：
Input shape：
Output shape：
MAE (u,v,w)：
RMSE (u,v,w)：
Prediction finite：
成功/失败：
若失败，报错与原因：
```

## 5. 后续实验安排

| 实验 | 目标 | 执行顺序 | 验收标准 |
| --- | --- | --- | --- |
| **E00 预训练推理** | GPU + 模型反序列化 + 数据格式核验 | **当前第一优先级** | 可以推理，输出有限，误差计算成功 |
| **E01 短训练 Smoke Test** | 验证 PDE 二阶自动微分、反向传播、优化器 | E00 通过后 | 10～50 次更新无 OOM / NaN，loss 可打印 |
| **E02 正式重训** | 固定环境、采样、种子，重跑 `data/100` | E01 通过后 | 训练过程、checkpoint、日志可复查 |
| **E03 采样密度对比** | 对比 `5` / `20` / `100` | E02 稳定后 | 相同独立评估协议、报告误差与成本 |
| **E04 PINN+时间序列** | 输入扩展 `(x,y,z,t)` 与非稳态 NS | 完成稳态 baseline 后 | 同时空数据上建立公平对比 |

### E01：短训练建议（尚未修改代码）

当前 `pinn/pinn_train.py` 中写死了：

```python
path_data = "../data/100/"
BATCH_SIZE_PHY = 31000
for epoch in range(500000):
    ...
```

第一次 smoke test 建议在**实验分支**调整为：

```python
path_data = "../data/5/"  # 可减少监督点；也可以仍用 100
BATCH_SIZE_PHY = 256       # 先检查显存与 PDE 自动微分
for epoch in range(20):    # 只验证流程，不代表收敛
    ...
```

然后在仓库根目录创建原脚本使用的**缺失保存目录**（短训练通常达不到第一次保存点，但正式重训必须创建）：

```bash
mkdir -p result_model_pinn
cd pinn
python pinn_train.py
```

请注意：以上是**修改代码后才执行的建议**，不是可以直接按原文件无修改完成的短训练命令。原始脚本每 50 步打印、每 1,000 步保存，20 步时不会自动打印损失；smoke test 需要把日志输出条件同时改小（例如每 1 步打印），并检查是否出现 `NaN/Inf`。

### E02/E03：正式实验必须明确控制变量

- 记录固定 Python/PyTorch/CUDA/驱动版本、Git commit、随机种子；
- 使用统一评价数据（最好独立高分辨率真值），避免把训练点拟合误差当作超分辨率效果；
- `5/20/100` 在源代码中并未明示为「采样率百分比」，**不要直接按百分比解释**；已知对应样本量分别为 334/2875/8297；
- 区分 `loss_data` 和 `loss_phy` 的变化，关注实际误差与时间、显存；
- 建议把 `step,seed,case,loss_data,loss_phy,loss_total,wall_time` 记录到日志或 CSV；
- 不要混用不同数据组对应的坐标归一化上下界与预训练权重。

## 6. 已发现的复现风险（优先处理）

| 级别 | 位置 | 原因 / 应对 |
| --- | --- | --- |
| P0 | `pinn/pinn_train.py` | 保存目标 `../result_model_pinn/` 不在原树中；重训前建目录 |
| P0 | `pinn/pinn.py` | `mse_data` 为 `abs((pred-true)/true)` 的求和；真实速度接近/等于 0 时可能 `Inf/NaN`。需要验证并单独提交数值稳定性修复 |
| P1 | `pinn/pinn.py` | 数据/物理残差都用 `sum` 而非 `mean`，采样点数变化会影响损失规模；做 E03 前必须界定指标归一化 |
| P1 | `pinn/pinn_predict_plane.py` | 顶层 `mayavi` 依赖可能阻塞无头服务器，首次诊断不要用它 |
| P1 | `pinn/pinn_predict_plane.py` | `flag=0/1` 的 x/y 截面选择未实现，原来的 `flag=2/3` 才有对应选择 |
| P1 | `pinn/pinn_predict_plane.py` | `txt_read()` 固定删除第一行，但已见样本 `content.txt` 第一行可能是数值，而不是表头 |
| P1 | checkpoint | `torch.save(pinn, ...)` 完整对象依赖 Python 类路径与 PyTorch 序列化版本；注意安全与兼容性 |
| P2 | 工程文件 | 原仓库缺少统一依赖、运行配置、忽略规则、独立评估脚本；尽量逐项增加，**不要在本次文档提交中擅自改动训练算法** |

**本次文档提交只增加实验说明**；不宣称上述风险已经修复，也不宣称已经完成任何 Linux 实测。

## 7. 如何提交后续实验记录

先让 E00 真正运行，再将第 4.4 节填写成真实结果。建议实验产生的模型、临时日志、可视化和大 CSV 先存服务器独立目录（例如 `~/pinn_sr_runs/`），避免在原仓库当前缺少 `.gitignore` 时意外提交大量结果。

推荐的实验命名（**新建议，不是上游既有规则**）：

```text
E00_pretrained_100
E01_smoke_5_seed1234
E02_train_100_seed1234
E03_sparse_5_20_100
```

复现过程中的每项变更应保留：对应代码 commit、数据组、环境、命令、指标、异常，以及「原实现 / 修复版」的差异。

## 8. 当前下一步行动

**现在只做以下三件事：**

1. 在 Linux 上运行 `nvidia-smi` 并记录 GPU/Driver/显存；
2. 创建 `pinn_sr` 环境，完成 `torch.cuda.is_available()` 检查；
3. 在 `pinn/` 目录运行 **E00 Pretrained Model Test**，保留完整输出。

若 E00 报错，先记录错误，不要直接开始 500,000 步训练；后续根据真实 GPU 和首次评估结果选择 E01 参数。

---

**关联文档**：[根目录 README：架构、数据文件命名和完整复现风险](../../README.md) · [训练入口](../../pinn/pinn_train.py) · [PINN 方程与损失](../../pinn/pinn.py) · [网络定义](../../pinn/network.py)。
