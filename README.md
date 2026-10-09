# PINN-Super-resolution：代码结构、命名规范与复现指南

> **仓库（个人 Fork）**：https://github.com/yigeyouzia/PINN-Super-resolution  
> **原始上游**：https://github.com/wanghuan10/PINN-Super-resolution  
> **分析对象**：Fork 的 `main` 分支，2026-10-09 所见代码  
> **文档性质**：基于实际源代码与部分 CSV 的静态核查；**尚未执行训练或运行预训练权重**。文中的「预期」不代表已复现的结果。

## 1. 一句话理解项目

这是一个**基于 PyTorch 的三维、稳态、不可压缩流场重建 PINN**。使用少量带速度标注的空间点和更多物理配点，让全连接网络学习连续映射：

```text
输入 (x, y, z) --> 全连接 DNN --> (u, v, w, p)
                  |                 |
             数据监督损失       NS 物理残差
                  \                 /
                   加权总损失 --> Adam
```

核心思想是通过物理约束，在没有速度标注的空间位置也可以预测流场。仓库标题中的 *Super-resolution* 主要体现为**从离散/稀疏观测学习连续空间场**，不是图像超分辨率网络，也不是 Transformer/FNO/时序模型。

**研究边界**：本仓库当前没有时间输入 `t`；训练的 `S=1` 不是实际的时间序列建模，物理残差也没有时间导数。用于「PINN + 时间序列」研究时应当视作**空间重建 baseline**。

## 2. 原仓库结构（保留原命名）

```text
PINN-Super-resolution/
├── data/
│   ├── 5/
│   ├── 20/
│   ├── 100/
│   ├── upper hole/
│   ├── lower hole/
│   ├── truncation/
│   └── phy.csv
├── pinn/
│   ├── network.py
│   ├── pinn.py
│   ├── pinn_train.py
│   ├── pinn_predict_plane.py
│   ├── Utilities/
│   │   └── plotting.py
│   └── __pycache__/               # 已被原仓库追踪的缓存
├── result_model_pinn_5/
│   └── model_pinn_196000
├── result_model_pinn_20/
│   └── model_pinn_198000
├── result_model_pinn_100/
│   └── model_pinn_197000
├── result_model_pinn_left/
│   └── model_pinn_200000
├── result_model_pinn_right/
│   └── model_pinn_195000
└── result_model_pinn_truncation/
    └── model_pinn_199000
```

**清单核对**：原始目录没有 `README.md`、`requirements.txt`、`environment.yml`、`setup.py`、`tests/` 和 `.gitignore`。本文件是 Fork 新增的导读，不代表上游原有说明。`.pyc` 被纳入版本控制，不建议未来继续追踪。

### 2.1 源文件职责

| 文件 | 入口 / 关键函数 | 实际职责 |
| --- | --- | --- |
| `pinn/network.py` | `layer`、`DNN`、`weights_init` | 构造 MLP、归一化输入坐标、Xavier 参数初始化，并设置 NumPy/PyTorch 随机种子 1234 |
| `pinn/pinn.py` | `PINN.predict`、`loss_func`、`closure` | 组装 DNN、数据损失、NS PDE 残差、可学习 PDE 系数及梯度反传 |
| `pinn/pinn_train.py` | `load_data`、`load_phy`、`training_data`、`__main__` | 加载监督/物理点，主训练循环，打印损失、保存完整对象 checkpoint |
| `pinn/pinn_predict_plane.py` | `txt_read`、`predict_plane`、`plane_vector`、`plane_scalar` | 载入已保存模型、截面/三维预测，Mayavi 矢量图和 Matplotlib 标量图 |
| `pinn/Utilities/plotting.py` | `newfig`、`savefig` | 通用 Matplotlib/LaTeX 绘图辅助；当前主训练与预测代码**没有显式导入它** |

源码入口：[network.py](pinn/network.py) · [pinn.py](pinn/pinn.py) · [pinn_train.py](pinn/pinn_train.py) · [pinn_predict_plane.py](pinn/pinn_predict_plane.py)。

## 3. 文件命名规则与数据约定

### 3.1 数据目录

`data/{case}/` 表示一组空间观测数据。实际存在的 `case` 值为：

- 数值组：`5`、`20`、`100`
- 场景组：`upper hole`、`lower hole`、`truncation`

**注意**：数字 `5/20/100` 不是代码声明的网格放大倍数，也没有明确的采样率说明。只能确认对应数据点数量不同；其科学含义需要原作者实验说明支持。

每组内包含以下文件：

| 文件名 | 已核实的用途 / 结构 |
| --- | --- |
| `coordinate.csv` | 无表头，`N × 3`，列依次为 `x, y, z` |
| `velocity_x.csv` | 无表头，`1 × N`，第 `i` 个值对应第 `i` 个坐标的 `u` |
| `velocity_y.csv` | 同理，对应 `v` |
| `velocity_z.csv` | 同理，对应 `w` |
| `velocity.csv` | 单行速度大小/派生速度标量；**现有训练器不读取**，不要误作三分量 |
| `content.csv` | 综合内容文件，至少部分场景为多列数值；**训练器不读取** |
| `content.txt` | 预测可视化器读取，取前 3 列坐标、接下来 3 列速度；各场景列数/内容可能不一致 |

另有根目录 `data/phy.csv`，由 `load_phy("../data/")` 加载，**至少**使用前三列 `(x,y,z)` 作为物理 collocation 点。它不属于任何 `data/{case}` 子目录。

### 3.2 实际 CSV 观测点数量

从 Fork 当前 `coordinate.csv` 完整内容计数得到：

| `data/{case}` | 坐标点数 N | `coordinate.csv` 形状 |
| --- | ---: | --- |
| `5` | 334 | `(334,3)` |
| `20` | 2,875 | `(2875,3)` |
| `100` | 8,297 | `(8297,3)` |
| `upper hole` | 8,075 | `(8075,3)` |
| `lower hole` | 7,233 | `(7233,3)` |
| `truncation` | 7,011 | `(7011,3)` |

已额外核对 `5` 的三份速度分量 CSV 均为 `1×334`，`20/100` 的 `velocity_x.csv` 分别为 `1×2875` 与 `1×8297`。代码里的 `.T` 把分量速度变为 `N×1`，再拼成 `(N,3)`。

**容易踩坑**：
1. `content.txt` 并不统一：`data/5/content.txt` 的示例行是 **7 个数值列**，而 `data/100/content.txt` 与其 `content.csv` 是同一个 Git blob，示例为 **9 列**；绘图只用前 6 列。
2. `txt_read()` 将 `array_data[1:, :]` 当作「去表头」，但已检查的 `content.txt` 首行本身就是数值，**会丢失第一个有效数据点**。
3. 文件路径中有空格，如 `upper hole`；写 Shell 命令时需要加引号。
4. `result_model_pinn_left/right` 并不能直接从源码证明与 `upper/lower hole` 的一一映射关系，**不可自行假定**。

### 3.3 模型输出命名

模式为 `result_model_pinn_{case}/model_pinn_{iteration}`，例如：

```text
result_model_pinn_100/model_pinn_197000
```

此处 `197000` 是文件命名中的训练迭代标记，文件**没有 `.pt` 或 `.pth` 扩展名**。代码使用 `torch.save(pinn, path)` 序列化整个 Python `PINN` 实例，**不是仅保存 `state_dict`**。

原 `pinn_train.py` 实际保存到 `../result_model_pinn/model_pinn_%d`（**无 `_100`**），而这个父目录在原始树中**不存在**；训练达到首次保存点前必须先创建目录，或修改输出路径与已有目录一致。

## 4. 网络、物理方程、损失及训练

### 4.1 网络结构（见 `pinn_train.py`）

```python
layers = [3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]
```

- 输入：3 维空间坐标 `(x, y, z)`
- 隐藏层：**10 层 × 100 单元**，`Tanh` 激活
- 输出：4 维 `(u, v, w, p)`，压力 `p` 没有独立监督标签
- 输入归一化：`(x - lb) / (ub - lb)`；`ub/lb` 来源于所选观测数据集坐标极值
- 线性层权重 Xavier uniform、偏置 0；`network.py` 中 NumPy/Torch 种子固定为 1234
- `PINN` 是普通 Python 类，内部 `self.net` 才是 PyTorch `nn.Module`

### 4.2 PDE 与可学习参数（见 `pinn.py`）

代码通过 `torch.autograd.grad(..., create_graph=True)` 求一阶、二阶空间导数。

不可压缩连续方程：

``math
f_c = u_x + v_y + w_z
``math

x 方向动量残差可等价写成：

``math
f_u = u u_x + v u_y + w u_z
      + \lambda_{1x} p_x
      - \lambda_{2x} \left[\nabla^2 u + \partial_x(u_x+v_y+w_z)\right].
``math

y、z 方向使用相同形式，分别对应 `\lambda_{1y/z}` 和 `\lambda_{2y/z}`；这与代码中的黏性二阶导数组合对应。这里**没有** `u_t/v_t/w_t`，因此是稳态方程实现。

额外可训练参数：`lambda_1x/y/z`（压力梯度系数）与 `lambda_2x/y/z`（黏性项系数）共 6 个。`alpha` 也被创建和注册为可训练参数，但**没有参与 `loss_func()` 计算**，因而没有实际损失梯度贡献。

### 4.3 损失函数

源码 `mse_data` 注释为「MRE」，真实形式是**未取平均的逐点绝对相对误差求和**，分别对三种速度分量求和：

``math
L_{\mathrm{data}} =
\sum_i \left|\frac{\hat u_i-u_i}{u_i}\right|
+\sum_i \left|\frac{\hat v_i-v_i}{v_i}\right|
+\sum_i \left|\frac{\hat w_i-w_i}{w_i}\right|.
``math

物理损失为连续方程与三个动量残差的平方**求和**：

``math
L_{\mathrm{phy}} = \sum_j(f_{c,j}^2+f_{u,j}^2+f_{v,j}^2+f_{w,j}^2).
``math

总损失：

``math
L = 5 L_{\mathrm{data}} + 1 L_{\mathrm{phy}}.
``math

**重要**：`u_i/v_i/w_i` 可能接近或等于 0，直接相除可能产生 `Inf/NaN`；代码虽然替换了输入/预测中的 `NaN`，但**没有保护相对误差分母**。而且 `sum`（非 `mean`）意味着更改数据点或配点数量会改变损失量级。做不同稀疏度实验前，必须统一损失归一化和数值策略。

### 4.4 训练超参数与循环

| 参数 | 当前硬编码值 |
| --- | --- |
| 训练观测集 | `../data/100/`（8,297 个坐标点） |
| 物理点 | `../data/phy.csv` |
| 物理批大小 | `BATCH_SIZE_PHY = 31000` |
| 训练总循环次数 | `range(500000)` |
| 优化器 | `torch.optim.Adam(self.net.parameters())`（默认学习率 `1e-3`） |
| 日志间隔 | 每 50 次迭代 |
| 模型保存间隔 | 每 1,000 次迭代 |
| 训练数据放置 | 全量有标签点移到 GPU/CPU 目标设备；物理配点先留在 CPU，再分批搬运 |

`closure()` 只进行 `zero_grad()`、损失计算、反向传播并返回指标；真正的参数更新 `pinn.adam.step()` 在训练脚本主循环中。

## 5. 复现流程：优先验证现有权重，再尝试训练

以下命令以 Linux/macOS 终端、仓库 `main` 为起点。**这是建议执行路径，不是已经通过测试的环境**。原仓库没有锁定版本，依赖兼容需要在实际机器验证。

### Step 0：克隆个人 Fork，独立环境

```bash
git clone https://github.com/yigeyouzia/PINN-Super-resolution.git
cd PINN-Super-resolution

# 可从 Python 3.10 环境开始尝试，不代表上游锁定版本
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scipy matplotlib
```

基础训练/数值预测使用以上 Python 依赖。**Mayavi 仅用于原 3D 可视化脚本**，如需要再安装 `mayavi`、`vtk` 和相应 Qt 图形环境；其安装可能比训练依赖复杂，特别是无图形界面服务器及部分 macOS 环境。

### Step 1：先做不依赖 Mayavi 的预训练模型推理（推荐）

仓库提供 `result_model_pinn_100/model_pinn_197000`。为了不被原预测脚本顶层 `from mayavi import mlab` 阻塞，可**仅用下面的独立诊断命令**读取 checkpoint 并计算训练观测点上的 RMSE。请在仓库根目录执行：

```bash
cd pinn
python - <<'PY'
import numpy as np
import torch

# 使 pickle 中的 PINN / DNN 类可从本目录解析
from pinn import PINN
from pinn_train import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "../result_model_pinn_100/model_pinn_197000"

# 仅对可信仓库的 checkpoint 使用 weights_only=False
model = torch.load(path, map_location=device, weights_only=False)
model.net.eval()

x, y, z, vx, vy, vz, *_ = load_data("../data/100/")
xyz = np.concatenate([x, y, z], axis=1)
target = np.concatenate([vx, vy, vz], axis=1)

with torch.no_grad():
    inputs = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    prediction = model.net(inputs)[:, :3].cpu().numpy()

rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=0))
print("n_points:", len(xyz))
print("RMSE u, v, w:", rmse)
print("all_finite:", np.isfinite(prediction).all())
PY
```

这个诊断是**同一观测集上的拟合误差**，**不是**真正的高分辨率独立测试集评估。不要把它当成论文的泛化/超分辨率精度。若无法成功 `torch.load`，先检查 PyTorch 版本与序列化类的兼容性。

**安全提示**：完整对象 pickle 可执行反序列化逻辑；`weights_only=False` 仅用于你信任的权重文件，不能用于任意来源文件。PyTorch 新版本默认采用更严格的 `weights_only` 行为，故这里显式指定。

### Step 2：使用原来的三维/截面绘图

原脚本文件：`pinn/pinn_predict_plane.py`，其 `__main__` 当前设置是：

```python
flag = 3                     # 全三维；flag=2 为 z 截面
file_name = "../data/100/content.txt"
model_sel = 197000
file_model = "../result_model_pinn_100/model_pinn_%d" % model_sel
fig_predicted = True
```

安装且配置成功 Mayavi/VTK/GUI 后，在 `pinn/` 目录执行：

```bash
python pinn_predict_plane.py
```

- `flag=3`：使用全部读入点显示 3D 矢量图。
- `flag=2`：按 `abs(z)` 选取 `plane ± 0.01` 截面。
- `flag=0/1`：虽然注释声称代表 x/y 截面，但 `predict_plane()` **没有对应选择逻辑**，原样执行会引用未定义的 `xyz_select_cpu`，需要先补实现。
- 标量绘图 `plane_scalar()` 的实际取值分支是 `flag=20` 或 `21`，和上面的截面选择标记是两套约定；调用前应检查 `val` 已赋值。

**不要在仓库根目录直接 `python pinn/pinn_predict_plane.py` 并假定相对路径自动正确**；当前实现依赖 `cwd=pinn/`。

### Step 3：尝试从头训练

训练前先建立实际保存目标目录，避免第 1,000 次迭代保存时报「目录不存在」：

```bash
# 回到仓库根目录
cd ..
mkdir -p result_model_pinn
cd pinn
python pinn_train.py
```

此命令按原始设置使用 `data/100`、全量监督点与物理 mini-batch，目标为 500,000 次 Adam 更新，计算与显存开销较大。**建议先在个人实验分支把迭代数、物理批大小、保存频率改小，进行 10~50 步 smoke test；不要将 smoke test 视为结果复现。**

观察 `Loss`、`mse_data`、`mse_phy` 是否均为有限数且总体有下降趋势；若遇 `nan/inf`，优先检查相对误差分母、梯度与损失尺度。训练完成后应区分「成功跑通」与「达到原实验精度」，还需要独立测试集、误差指标和随机种子对照。

### Step 4：切换其他数据组

修改 `pinn/pinn_train.py` 的 `training_data()` 中：

```python
path_data = "../data/5/"    # 也可指定 20、100、truncation 等
```

并将保存目录设为专属 `result_model_pinn_{case}` 或统一 `runs/{case}/`。使用现有 checkpoint 预测时要相应更改 `file_name` 与 `file_model`，**不要混用数据组的坐标归一化范围与模型权重**。不同场景需核实对应训练/评估协议。

## 6. 已识别的主要复现风险与修复优先级

| 优先级 | 文件位置 | 问题与推荐处理 |
| --- | --- | --- |
| **P0** | `pinn_train.py` | 保存 `../result_model_pinn/`，但原仓库未创建此目录；预先 `mkdir -p` 或修改保存逻辑 |
| **P0** | `pinn.py` | 数据损失对速度真实值直接相除，接近/等于零不稳定；建议加 `epsilon` 或选用带量纲归一化 MAE/MSE，并记录改动对实验的影响 |
| **P0** | `pinn_predict_plane.py` | `torch.load` 完整 pickle 与新版本默认 `weights_only` 可能冲突；只对可信 checkpoint 显式处理版本和加载策略 |
| **P1** | 多脚本 | 相对路径依赖工作目录，建议改成 `pathlib.Path(__file__).resolve()` 定位项目根目录 |
| **P1** | `pinn_predict_plane.py` | `flag=0/1` 代码未实现；`txt_read` 对无表头 TXT 错删首行；两处均应修复后再做定量分析 |
| **P1** | `pinn.py` | 数据损失和物理损失均用 `sum`，不同 N/mini-batch 导致损失尺度不可直接比较；建议改为 `mean` 或规范权重 |
| **P1** | `pinn_train.py` | 迭代数、输入场景、物理批大小、权重全部硬编码；建议 `argparse/YAML` 配置 |
| **P1** | `pinn_predict_plane.py` | 顶层硬依赖 Mayavi，服务器无图形环境无法直接运行；建议分离纯数值预测与可视化 |
| **P1** | 项目整体 | 没有独立测试/误差评估脚本、依赖锁定文件和训练日志落盘；先建立可检查的定量协议 |
| **P2** | `pinn.py` | `alpha` 注册可训练但不参与损失；无用参数应删除或明确用途 |
| **P2** | `pinn_train.py` | 物理点顺序批切 + 尾批大小可能变化；建议验证采样覆盖与改为可复现 DataLoader |
| **P2** | 全仓库 | 已追踪 `__pycache__/*.pyc`，缺少 `.gitignore`；建议后续清理生成文件并忽略实验输出 |

**变更原则**：本次只添加文档，**没有擅自修改原训练、损失、模型及数据**。上述修复一旦实施都应单独提交，并说明与原方法的差异。

## 7. 建议的实验记录/文件命名标准（新规范，不是原仓库约定）

为了后续与 PINNsFormer、ConFIG、时序模型统一比较，建议后续将每次实验按「case / seed / 配置」组织：

```text
configs/
  pinn_ns3d_5.yaml
  pinn_ns3d_20.yaml
  pinn_ns3d_100.yaml
runs/
  ns3d_5_seed1234/
    config.yaml
    train_metrics.csv
    checkpoint_stepXXXXXX.pt
    eval_metrics.json
    predicted_field.npz
docs/
  methods/
  experiments/
```

建议区分：
- `case`：数据来源（如 `5`、`100` 或特殊场景）
- `seed`：随机种子
- `step`：优化更新次数，不混用 `epoch`
- `train_metrics.csv`：`step,loss_total,loss_data,loss_phy,elapsed_sec` 等字段
- `eval_metrics.json`：独立评估数据、RMSE/相对误差、数据点数与坐标域说明
- `.pt`：优先保存 `state_dict`、优化器状态、配置、步数、运行环境，而不是 pickle 整个类

优先级建议：**先以现有权重完成可重复预测 → 修 P0 问题并跑通短训练 → 定义独立场重建评估 → 再做不同采样密度的受控对比。**

## 8. 与未来「PINN + 时间序列」课题的关系

**可以直接借鉴**：空间坐标归一化、`u/v/w/p` 多输出场回归、自动微分物理残差、稀疏测点与 PDE collocation 的联合训练、不同观测密度下的实验协议。

**但还需新建的能力**：

1. 将输入改为 `(x,y,z,t)`，按真实时间戳组织观测；区分空间稀疏、时间稀疏和联合稀疏。
2. 将稳态动量方程扩展为非稳态形式，加入 `u_t/v_t/w_t`，并明确边界条件、初始条件、流体参数、单位/无量纲化。
3. 选择时间建模方法：基础 PINN-MLP → 时间编码 / Transformer / Mamba / SSM；先保持数据/参数/物理约束对齐，再公平比较。
4. 建立独立时空场真值和指标，分别报告时间插值、时间外推、空间超分辨和推理成本。
5. 若研究多任务损失冲突，可在**统一物理方程和采样协议**的前提下对比固定权重、梯度平衡或 ConFIG 等方法；当前原仓库只采用固定 `5:1` 权重，没有实现这些算法。

### 最小研究里程碑

- [ ] A：在已知环境里加载 `model_pinn_197000`，完成数值预测并确认结果有限
- [ ] B：保存数据形状、依赖版本、环境和可运行命令
- [ ] C：修复 P0、P1 并完成可复现的短训练
- [ ] D：构建**独立**测试场与统一评估指标，复现 5/20/100 空间稀疏度对比
- [ ] E：引入真实时间维度，建立时空 PINN baseline，再接入时间序列结构

---

## 9. 阅读和核查依据

本导读直接阅读了如下 5 个 Python 文件：
- [`pinn/network.py`](pinn/network.py)
- [`pinn/pinn.py`](pinn/pinn.py)
- [`pinn/pinn_train.py`](pinn/pinn_train.py)
- [`pinn/pinn_predict_plane.py`](pinn/pinn_predict_plane.py)
- [`pinn/Utilities/plotting.py`](pinn/Utilities/plotting.py)

同时枚举了整个仓库文件树，并检查了上述场景的 `coordinate.csv`、部分速度分量 CSV 与 `content` 样本。没有验证上游论文中具体实验数值；没有执行 Mayavi 可视化、训练、完整数值复现；`data/phy.csv` 的总行数及特殊模型与场景之间的映射也未独立确认。因此，本文所有「修复」「实验」「建议」都是**代码审阅建议**，而非已经验证的复现实验结果。
