"""
互信息估计实验：MINE (Mutual Information Neural Estimation)
基于已缓存的各层特征 (.npz) 直接运行，无需重新过模型

实验目的：
  区分"Attr信息真的不在浅层" vs "浅层有但非线性编码"
  Linear Probing 只能捕捉线性可分性，MINE 能捕捉任意非线性依赖

方法：
  MINE: I(X;Y) = E_joint[T(x,y)] - log(E_marginal[e^T(x,y)])
  其中 T 是可学习的统计量网络 (Statistics Network)

输出：
  - 每层的 MI(feature, attr) 和 MI(feature, obj) 估计值
  - 与 Linear Probing 结果的对比图（揭示非线性信息）
  - 非线性增益曲线 (MI - LinearProbe_Acc 的相对差)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
from tqdm import tqdm

# ==========================================
# 0. 配置
# ==========================================
CFG = {
    "feature_cache_dir": "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Point2/code/feature_cache",
    "output_dir":        "./mi_results",
    "num_layers":        24,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",

    # MINE 训练超参
    "mine_hidden_dim":   512,       # 统计量网络隐层维度
    "mine_epochs":       300,       # 训练轮数
    "mine_batch_size":   2048,
    "mine_lr":           1e-4,
    "mine_ema_decay":    0.01,      # EMA 系数，用于稳定对数期望项
    "early_stop_patience": 30,

    # PCA 降维（MINE 对高维输入敏感，建议先降维）
    "pca_dim":           64,        # 降到多少维；None = 不降维

    # 是否加载之前 probing 的结果用于对比
    "probing_result_path": "./probing_results/probing_results.json",

    # 随机种子
    "seed": 42,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ==========================================
# 1. MINE 核心实现
# ==========================================
class StatisticsNetwork(nn.Module):
    """
    MINE 的统计量网络 T(x, y)。
    输入：特征 x 与标签 embedding y 的拼接
    输出：标量得分

    使用 ELU 激活而非 ReLU，避免梯度消失。
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # 初始化最后一层接近零，使训练初期稳定
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy).squeeze(-1)


class MINEEstimator:
    """
    MINE 估计器。
    
    损失函数（最大化下界）：
        L = E_joint[T] - log(E_marginal[exp(T)])
    
    其中 marginal 通过打乱 y 的 batch 顺序来近似。
    使用 EMA 版本稳定训练（MINE-f 变体）。
    """
    def __init__(self, x_dim: int, y_dim: int, cfg: dict):
        self.device = cfg["device"]
        self.net = StatisticsNetwork(
            x_dim, y_dim, cfg["mine_hidden_dim"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=cfg["mine_lr"]
        )
        self.ema_decay = cfg["mine_ema_decay"]
        self.ema_et    = 1.0   # EMA 滑动估计，初始为1
        self.history   = []

    def _mine_loss(self, x: torch.Tensor, y: torch.Tensor):
        """
        返回 -MI 下界（取负是因为我们用梯度下降，要最小化）
        使用 EMA 稳定 log(E[exp(T)]) 项
        """
        # 联合分布样本
        t_joint = self.net(x, y)

        # 边缘分布样本：打乱 y 的顺序使 x, y 独立
        y_shuffle = y[torch.randperm(y.size(0), device=self.device)]
        t_marginal = self.net(x, y_shuffle)

        # EMA 稳定版 log E[exp(T)]
        et = torch.exp(t_marginal).mean().item()
        self.ema_et = (1 - self.ema_decay) * self.ema_et + self.ema_decay * et
        
        # MINE-f 损失（梯度不穿过 EMA 项，但 log 用 EMA 值）
        loss = -(t_joint.mean() - (1.0 / self.ema_et) * torch.exp(t_marginal).mean())
        return loss

    def fit(self, X: np.ndarray, Y_embed: np.ndarray,
            epochs: int, batch_size: int, patience: int) -> float:
        """
        训练 MINE 并返回最终 MI 估计值（nats）。
        
        Args:
            X:       特征矩阵，[N, D]
            Y_embed: 标签的 embedding，[N, E]（one-hot 或 learned）
        """
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(Y_embed, dtype=torch.float32).to(self.device)
        N = X_t.size(0)

        best_mi    = -float("inf")
        best_state = None
        patience_cnt = 0
        mi_history = []

        for epoch in range(epochs):
            # Mini-batch 训练
            perm = torch.randperm(N, device=self.device)
            epoch_losses = []
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                bx, by = X_t[idx], Y_t[idx]
                
                self.optimizer.zero_grad()
                loss = self._mine_loss(bx, by)
                loss.backward()
                # 梯度裁剪，防止爆炸
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                self.optimizer.step()
                epoch_losses.append(-loss.item())
            
            # 用全量数据估计当前 MI
            with torch.no_grad():
                t_joint    = self.net(X_t, Y_t).mean().item()
                y_shuf     = Y_t[torch.randperm(N, device=self.device)]
                t_marginal = torch.log(torch.exp(self.net(X_t, y_shuf)).mean()).item()
                mi_est     = t_joint - t_marginal
            
            mi_history.append(mi_est)
            self.history.append(mi_est)

            # 早停
            if mi_est > best_mi:
                best_mi  = mi_est
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        
        return best_mi


# ==========================================
# 2. 标签编码：one-hot
# ==========================================
def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """将整数标签转为 one-hot 向量。"""
    N = len(labels)
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    onehot[np.arange(N), labels] = 1.0
    return onehot


# ==========================================
# 3. PCA 降维（可选，减少 MINE 输入维度）
# ==========================================
def fit_pca(X_train: np.ndarray, n_components: int):
    """用 numpy SVD 实现 PCA，返回投影矩阵。"""
    mu = X_train.mean(axis=0, keepdims=True)
    X_centered = X_train - mu
    # 用随机化 SVD 加速
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_centered)
    return mu, pca

def apply_pca(X: np.ndarray, mu: np.ndarray, pca) -> np.ndarray:
    return pca.transform(X - mu).astype(np.float32)


# ==========================================
# 4. 加载缓存特征
# ==========================================
def load_layer_features(layer_id: int, split: str, cache_dir: str):
    """
    加载指定层、指定 split 的特征缓存。
    文件命名规范：layer{i}_{split}_normalized.npz
    """
    path = os.path.join(cache_dir, f"layer{layer_id}_{split}_normalized.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"请先运行 probing 实验生成特征缓存。"
        )
    data = np.load(path)
    return data["features"], data["attrs"], data["objs"]


# ==========================================
# 5. 主流程
# ==========================================
def main():
    device = CFG["device"]
    print(f"Device: {device}")
    print(f"PCA dim: {CFG['pca_dim']} ({'enabled' if CFG['pca_dim'] else 'disabled'})")

    # 先加载 Layer 0 获取元信息
    print("\n>>> 1. Loading metadata from Layer 0...")
    tr_feat0, tr_attrs, tr_objs = load_layer_features(0, "train", CFG["feature_cache_dir"])
    te_feat0, te_attrs, te_objs = load_layer_features(0, "test",  CFG["feature_cache_dir"])

    num_attrs   = max(tr_attrs.max(), te_attrs.max()) + 1
    num_objs    = max(tr_objs.max(),  te_objs.max())  + 1
    feat_dim_raw = tr_feat0.shape[1]
    print(f"  Train: {len(tr_attrs)}, Test: {len(te_attrs)}")
    print(f"  #Attrs: {num_attrs}, #Objs: {num_objs}, Raw feat dim: {feat_dim_raw}")

    # 准备标签 one-hot（所有层共用）
    tr_attr_oh = to_onehot(tr_attrs, num_attrs)
    tr_obj_oh  = to_onehot(tr_objs,  num_objs)
    te_attr_oh = to_onehot(te_attrs, num_attrs)
    te_obj_oh  = to_onehot(te_objs,  num_objs)
    print(f"  One-hot dims: attr={num_attrs}, obj={num_objs}")

    # ==========================================
    # 逐层估计互信息
    # ==========================================
    print(f"\n>>> 2. Running MINE for {CFG['num_layers']} layers...")
    
    mi_attr_list = []
    mi_obj_list  = []
    
    for layer_id in range(CFG["num_layers"]):
        print(f"\n  [Layer {layer_id:2d}]")
        
        # 加载特征
        tr_feat, _, _ = load_layer_features(layer_id, "train", CFG["feature_cache_dir"])
        te_feat, _, _ = load_layer_features(layer_id, "test",  CFG["feature_cache_dir"])
        
        # PCA 降维
        if CFG["pca_dim"] and CFG["pca_dim"] < tr_feat.shape[1]:
            mu, pca = fit_pca(tr_feat, CFG["pca_dim"])
            tr_feat_pca = apply_pca(tr_feat, mu, pca)
            te_feat_pca = apply_pca(te_feat, mu, pca)
            feat_dim = CFG["pca_dim"]
        else:
            tr_feat_pca = tr_feat
            te_feat_pca = te_feat
            feat_dim = tr_feat.shape[1]
        
        y_dim_attr = num_attrs
        y_dim_obj  = num_objs
        
        # --- 估计 MI(feature, attr) ---
        print(f"    MI(feat, attr): training MINE... ", end="", flush=True)
        mine_attr = MINEEstimator(feat_dim, y_dim_attr, CFG)
        mi_attr = mine_attr.fit(
            tr_feat_pca, tr_attr_oh,
            epochs=CFG["mine_epochs"],
            batch_size=CFG["mine_batch_size"],
            patience=CFG["early_stop_patience"],
        )
        print(f"MI = {mi_attr:.4f} nats")
        
        # --- 估计 MI(feature, obj) ---
        print(f"    MI(feat, obj):  training MINE... ", end="", flush=True)
        mine_obj = MINEEstimator(feat_dim, y_dim_obj, CFG)
        mi_obj = mine_obj.fit(
            tr_feat_pca, tr_obj_oh,
            epochs=CFG["mine_epochs"],
            batch_size=CFG["mine_batch_size"],
            patience=CFG["early_stop_patience"],
        )
        print(f"MI = {mi_obj:.4f} nats")
        
        mi_attr_list.append(float(mi_attr))
        mi_obj_list.append(float(mi_obj))
        
        # 实时保存（防止中途崩溃丢失）
        interim = {
            "completed_layers": layer_id + 1,
            "mi_attr": mi_attr_list,
            "mi_obj":  mi_obj_list,
        }
        with open(os.path.join(CFG["output_dir"], "mi_results_interim.json"), "w") as f:
            json.dump(interim, f, indent=2)

    # ==========================================
    # 6. 保存最终结果
    # ==========================================
    layer_indices = list(range(CFG["num_layers"]))
    mi_gap = [abs(a - o) for a, o in zip(mi_attr_list, mi_obj_list)]

    results = {
        "layer_indices": layer_indices,
        "mi_attr":       mi_attr_list,
        "mi_obj":        mi_obj_list,
        "mi_gap":        mi_gap,
        "config":        {k: v for k, v in CFG.items()
                          if k not in ("feature_cache_dir",)},
    }
    result_path = os.path.join(CFG["output_dir"], "mi_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  MI results saved → {result_path}")

    # ==========================================
    # 7. 可视化
    # ==========================================
    print("\n>>> 3. Plotting...")
    plot_mi_results(layer_indices, mi_attr_list, mi_obj_list)
    print("\n✅ Done!")


# ==========================================
# 6. 可视化
# ==========================================
def plot_mi_results(layer_indices, mi_attr, mi_obj):
    """
    四子图可视化：
      1. MI 曲线（主图，含双轴 MI Gap）
      2. MI 逐层增益
      3. 与 Linear Probing 对比（若结果文件存在）
      4. 非线性信息增益：MI - normalized(LinearAcc)
    """
    L = layer_indices
    mi_attr = np.array(mi_attr)
    mi_obj  = np.array(mi_obj)
    mi_gap  = np.abs(mi_attr - mi_obj)

    # 尝试加载 probing 结果
    probing_data = None
    if os.path.exists(CFG["probing_result_path"]):
        with open(CFG["probing_result_path"]) as f:
            probing_data = json.load(f)
        print(f"  Loaded probing results for comparison.")

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ── 子图 1：MI 主曲线 ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(L, mi_attr, 'o-', color='#1565C0', lw=2.5, ms=7,
             label='MI(feature, Attribute)')
    ax1.plot(L, mi_obj,  's-', color='#B71C1C', lw=2.5, ms=7,
             label='MI(feature, Object)')

    # 标注关键层（MI 一阶差分最大处）
    attr_gain_mi = np.diff(mi_attr, prepend=mi_attr[0])
    obj_gain_mi  = np.diff(mi_obj,  prepend=mi_obj[0])
    key_attr = int(np.argmax(attr_gain_mi))
    key_obj  = int(np.argmax(obj_gain_mi))
    ax1.axvline(key_attr, color='#1565C0', ls=':', alpha=0.6,
                label=f'Max Attr MI Gain: Layer {key_attr}')
    ax1.axvline(key_obj,  color='#B71C1C', ls=':', alpha=0.6,
                label=f'Max Obj MI Gain: Layer {key_obj}')

    ax1r = ax1.twinx()
    ax1r.fill_between(L, mi_gap, alpha=0.12, color='#6A1B9A')
    ax1r.plot(L, mi_gap, '^--', color='#6A1B9A', lw=1.5, ms=5, alpha=0.75,
              label='MI Gap |Attr - Obj|')
    ax1r.set_ylabel('MI Gap (nats)', fontsize=11, color='#6A1B9A')
    ax1r.tick_params(axis='y', colors='#6A1B9A')

    ax1.set_title('MINE Mutual Information Estimate across ViT Layers\n(MIT-States, ViT-L/14)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('ViT Layer Index (Shallow → Deep)', fontsize=12)
    ax1.set_ylabel('Mutual Information I(X;Y) (nats)', fontsize=12)
    ax1.set_xticks(L)
    ax1.grid(True, ls='--', alpha=0.4)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=10, loc='upper left')

    # ── 子图 2：MI 逐层增益 ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    width = 0.38
    x = np.array(L)
    b1 = ax2.bar(x - width/2, attr_gain_mi, width, color='#42A5F5', alpha=0.85,
                 label='Attr MI Gain')
    b2 = ax2.bar(x + width/2, obj_gain_mi,  width, color='#EF5350', alpha=0.85,
                 label='Obj MI Gain')
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_title('Layer-wise MI Gain (MINE)', fontsize=12)
    ax2.set_xlabel('Layer Index', fontsize=11)
    ax2.set_ylabel('ΔMI (nats)', fontsize=11)
    ax2.set_xticks(L)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', ls='--', alpha=0.4)

    # ── 子图 3 & 4：与 Linear Probing 对比 ────────────────────
    if probing_data is not None:
        lp_attr = np.array(probing_data["attr_accuracies"])
        lp_obj  = np.array(probing_data["obj_accuracies"])
        
        # 归一化到 [0,1] 以便在同一坐标轴比较
        def norm01(x):
            xmin, xmax = x.min(), x.max()
            return (x - xmin) / (xmax - xmin + 1e-8)
        
        mi_attr_n = norm01(mi_attr)
        mi_obj_n  = norm01(mi_obj)
        lp_attr_n = norm01(lp_attr)
        lp_obj_n  = norm01(lp_obj)
        
        # 非线性信息增益 = MINE(归一化) - LinearProbe(归一化)
        # 正值 → 该层存在非线性编码的信息（线性探针捕捉不到）
        nonlinear_attr = mi_attr_n - lp_attr_n
        nonlinear_obj  = mi_obj_n  - lp_obj_n

        # 子图3：归一化对比曲线
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(L, mi_attr_n,  'o-',  color='#1565C0', lw=2,   ms=6, label='MINE Attr (norm)')
        ax3.plot(L, lp_attr_n,  'o--', color='#90CAF9', lw=1.5, ms=5, label='Linear Attr (norm)')
        ax3.plot(L, mi_obj_n,   's-',  color='#B71C1C', lw=2,   ms=6, label='MINE Obj (norm)')
        ax3.plot(L, lp_obj_n,   's--', color='#EF9A9A', lw=1.5, ms=5, label='Linear Obj (norm)')
        
        # 填充两种方法之间的差距（非线性信息区域）
        ax3.fill_between(L, lp_attr_n, mi_attr_n,
                         where=(mi_attr_n > lp_attr_n),
                         alpha=0.2, color='#1565C0', label='Nonlinear Attr Info')
        ax3.fill_between(L, lp_obj_n, mi_obj_n,
                         where=(mi_obj_n > lp_obj_n),
                         alpha=0.2, color='#B71C1C', label='Nonlinear Obj Info')
        
        ax3.set_title('MINE vs Linear Probing\n(Normalized for Comparison)', fontsize=12)
        ax3.set_xlabel('Layer Index', fontsize=11)
        ax3.set_ylabel('Normalized Score', fontsize=11)
        ax3.set_xticks(L)
        ax3.legend(fontsize=8, loc='upper left', ncol=2)
        ax3.grid(True, ls='--', alpha=0.4)

        # 保存对比数据
        results_with_comparison = {
            "layer_indices":    L,
            "mi_attr":          mi_attr.tolist(),
            "mi_obj":           mi_obj.tolist(),
            "mi_gap":           mi_gap.tolist(),
            "nonlinear_gain_attr": nonlinear_attr.tolist(),
            "nonlinear_gain_obj":  nonlinear_obj.tolist(),
            "interpretation": {
                "max_nonlinear_attr_layer": int(np.argmax(nonlinear_attr)),
                "max_nonlinear_obj_layer":  int(np.argmax(nonlinear_obj)),
                "max_mi_gap_layer":         int(np.argmax(mi_gap)),
                "note": (
                    "nonlinear_gain > 0 表示该层存在线性探针无法捕捉的非线性编码信息。"
                    "MI Gap 在某层显著增大，意味着 Object/Attr 信息在此层开始分化。"
                )
            }
        }
        with open(os.path.join(CFG["output_dir"], "mi_vs_linear_probe.json"), "w") as f:
            json.dump(results_with_comparison, f, indent=2, default=float)
        print(f"  Comparison data saved.")

    else:
        # 若无 probing 结果，子图3显示 MI 归一化曲线
        ax3 = fig.add_subplot(gs[1, 1])
        mi_attr_n = (mi_attr - mi_attr.min()) / (mi_attr.ptp() + 1e-8)
        mi_obj_n  = (mi_obj  - mi_obj.min())  / (mi_obj.ptp()  + 1e-8)
        ax3.plot(L, mi_attr_n, 'o-', color='#1565C0', lw=2, ms=6, label='MI Attr (norm)')
        ax3.plot(L, mi_obj_n,  's-', color='#B71C1C', lw=2, ms=6, label='MI Obj (norm)')
        ax3.set_title('Normalized MI curves', fontsize=12)
        ax3.set_xlabel('Layer Index', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, ls='--', alpha=0.4)

    save_path = os.path.join(CFG["output_dir"], "mi_estimation_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


# ==========================================
# 辅助：仅绘图（已有 mi_results.json 时跳过训练直接出图）
# ==========================================
def plot_only():
    """
    若 MINE 训练已完成，仅用保存的 JSON 重新绘图，无需重跑。
    用法：将 main() 改为 plot_only() 后运行。
    """
    result_path = os.path.join(CFG["output_dir"], "mi_results.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"MI results not found: {result_path}")
    
    with open(result_path) as f:
        data = json.load(f)
    
    plot_mi_results(data["layer_indices"], data["mi_attr"], data["mi_obj"])
    print("Plot regenerated.")


if __name__ == "__main__":
    # 正常运行：训练 MINE + 绘图
    main()
    
    # 若已有结果只需重新绘图：注释上面一行，取消下面注释
    # plot_only()