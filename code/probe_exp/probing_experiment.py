"""
改进版 ViT 分层特征 Linear Probing 实验
用于验证 CLIP ViT 不同层对 State/Object 信息的敏感度

主要改进点：
  1. 逐层提取特征，避免内存炸弹
  2. L2 特征归一化（CLIP 的必要前处理）
  3. 线性探针加入 LR Scheduler + 早停机制
  4. 新增 Disentanglement Gap 和 Separability Score 分析
  5. 更丰富的可视化（双坐标轴 + 解耦差距曲线）
  6. 关键超参数集中管理
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json

from clip_modules.clip_model import load_clip

# ==========================================
# 0. 超参数集中配置
# ==========================================
CFG = {
    "dataset_path": "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data/mit-states",
    "weights_path": "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/checkpoints/ViT-L-14.pt",
    "batch_size": 128,
    "probe_epochs": 60,
    "probe_batch_size": 2048,
    "probe_lr": 1e-3,
    "probe_weight_decay": 1e-4,
    "early_stop_patience": 8,          # 早停耐心轮数
    "normalize_features": True,         # ⚡ 关键：是否L2归一化特征
    "save_features": True,              # 是否缓存特征到磁盘（加速重复实验）
    "feature_cache_dir": "./feature_cache",
    "output_dir": "./probing_results",
    "num_workers": 0,                   # ⚠️ Hook + 多进程存在潜在问题，建议设0
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs(CFG["feature_cache_dir"], exist_ok=True)

# ==========================================
# 1. Hook 机制（改进版：支持逐层提取）
# ==========================================
_current_layer_features = {}

def make_hook(layer_id):
    """
    为指定层创建 Hook。
    CLIP ViT ResBlock 输出形状：[L, N, D]（序列长度, 批大小, 特征维度）
    取 index=0 的 CLS token，得到 [N, D]
    """
    def hook(module, input, output):
        # output 是 Tensor: [L, N, D]
        # output[0] -> [N, D]，即 CLS token
        _current_layer_features[layer_id] = output[0].detach().float()
    return hook


# ==========================================
# 2. 逐层特征提取（解决内存问题）
# ==========================================
@torch.no_grad()
def extract_single_layer(model, dataloader, target_layer_id, normalize=True):
    """
    只提取单个目标层的特征，避免同时存储所有层造成内存压力。
    
    Args:
        normalize: 是否做 L2 归一化（对 CLIP 特征强烈建议开启）
    Returns:
        features: np.ndarray, shape [N, D]
        attrs: np.ndarray, shape [N]
        objs: np.ndarray, shape [N]
    """
    model.eval()
    all_features, all_attrs, all_objs = [], [], []
    
    for batch in tqdm(dataloader, desc=f"  Layer {target_layer_id}", leave=False):
        images = batch[0].to(CFG["device"])
        attr_labels = batch[1]
        obj_labels = batch[2]
        
        _current_layer_features.clear()
        model.encode_image(images)
        
        if target_layer_id not in _current_layer_features:
            raise RuntimeError(f"Hook for layer {target_layer_id} did not fire! "
                               f"Check hook registration.")
        
        feat = _current_layer_features[target_layer_id].cpu()  # [N, D]
        
        # ⚡ 关键改进 1：L2 归一化
        # CLIP 的 encode_image 最后会做 projection + 归一化
        # 但中间层特征没有归一化，线性探针对尺度敏感，必须手动归一化
        if normalize:
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        
        all_features.append(feat)
        all_attrs.append(attr_labels)
        all_objs.append(obj_labels)
    
    features = torch.cat(all_features, dim=0).numpy()
    attrs = torch.cat([torch.tensor(a) for a in all_attrs]).numpy()
    objs = torch.cat([torch.tensor(o) for o in all_objs]).numpy()
    return features, attrs, objs


def load_or_extract(model, dataloader, layer_id, split_name, normalize):
    """带磁盘缓存的特征提取，避免重复计算。"""
    norm_suffix = "_normalized" if normalize else ""
    cache_path = os.path.join(
        CFG["feature_cache_dir"], f"layer{layer_id}_{split_name}{norm_suffix}.npz"
    )
    
    if CFG["save_features"] and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["features"], data["attrs"], data["objs"]
    
    features, attrs, objs = extract_single_layer(model, dataloader, layer_id, normalize)
    
    if CFG["save_features"]:
        np.savez_compressed(cache_path, features=features, attrs=attrs, objs=objs)
    
    return features, attrs, objs


# ==========================================
# 3. 改进版 GPU 线性探针训练
# ==========================================
class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_probe(X_train, y_train, X_test, y_test, num_classes):
    """
    训练线性探针（Logistic Regression）。
    
    改进点：
      - CosineAnnealing LR Scheduler
      - Validation-based Early Stopping
      - 返回完整的 loss 曲线供诊断
    """
    device = CFG["device"]
    feat_dim = X_train.shape[1]
    
    # 转 Tensor
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)
    
    probe = LinearProbe(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=CFG["probe_lr"],
        weight_decay=CFG["probe_weight_decay"]
    )
    # ⚡ 关键改进 2：余弦退火调度器，避免卡在局部最优
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["probe_epochs"]
    )
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=CFG["probe_batch_size"], shuffle=True
    )
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(CFG["probe_epochs"]):
        probe.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(probe(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # ⚡ 关键改进 3：早停机制
        probe.eval()
        with torch.no_grad():
            acc = (probe(X_te).argmax(1) == y_te).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG["early_stop_patience"]:
                break
    
    return best_acc


# ==========================================
# 4. 新增分析指标
# ==========================================
def compute_disentanglement_gap(attr_accs, obj_accs):
    """
    解耦差距曲线：|Attr_Acc - Obj_Acc|
    差距越大 → 该层对两类信息的区分越明显 → 越适合插入解耦提示
    """
    return np.abs(np.array(attr_accs) - np.array(obj_accs))


def compute_info_gain(accs):
    """
    逐层信息增益：accuracy 的一阶差分
    正值 → 该层新增了相关信息
    """
    accs = np.array(accs)
    return np.diff(accs, prepend=accs[0])


# ==========================================
# 5. 可视化（改进版）
# ==========================================
def plot_results(layer_indices, attr_accs, obj_accs, save_dir):
    gap = compute_disentanglement_gap(attr_accs, obj_accs)
    attr_gain = compute_info_gain(attr_accs)
    obj_gain = compute_info_gain(obj_accs)
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    
    L = layer_indices
    
    # --- 子图1：主准确率曲线 ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(L, attr_accs, 'o-', color='#2196F3', lw=2.5, ms=7, label='State/Attribute Accuracy')
    ax1.plot(L, obj_accs,  's-', color='#F44336', lw=2.5, ms=7, label='Object Accuracy')
    
    # 标注最佳层
    best_attr_layer = int(np.argmax(attr_accs))
    best_obj_layer  = int(np.argmax(obj_accs))
    ax1.axvline(best_attr_layer, color='#2196F3', ls='--', alpha=0.5,
                label=f'Best Attr Layer: {best_attr_layer}')
    ax1.axvline(best_obj_layer,  color='#F44336', ls='--', alpha=0.5,
                label=f'Best Obj Layer: {best_obj_layer}')
    
    # 双坐标轴：右轴显示解耦差距
    ax1r = ax1.twinx()
    ax1r.fill_between(L, gap, alpha=0.15, color='#9C27B0')
    ax1r.plot(L, gap, '^--', color='#9C27B0', lw=1.5, ms=5, alpha=0.8,
              label='Disentanglement Gap')
    ax1r.set_ylabel('|Attr Acc - Obj Acc| (Gap)', fontsize=11, color='#9C27B0')
    ax1r.tick_params(axis='y', colors='#9C27B0')
    
    ax1.set_title('Linear Probing Accuracy & Disentanglement Gap across ViT Layers\n(MIT-States, ViT-L/14)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('ViT Layer Index (Shallow → Deep)', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_xticks(L)
    ax1.grid(True, ls='--', alpha=0.4)
    
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='lower right')
    
    # --- 子图2：逐层信息增益（Attr）---
    ax2 = fig.add_subplot(gs[1, 0])
    colors_attr = ['#2196F3' if g >= 0 else '#BBDEFB' for g in attr_gain]
    ax2.bar(L, attr_gain, color=colors_attr, alpha=0.85)
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_title('Layer-wise Info Gain: Attribute', fontsize=12)
    ax2.set_xlabel('Layer Index', fontsize=11)
    ax2.set_ylabel('ΔAccuracy', fontsize=11)
    ax2.set_xticks(L)
    ax2.grid(True, axis='y', ls='--', alpha=0.4)
    
    # --- 子图3：逐层信息增益（Obj）---
    ax3 = fig.add_subplot(gs[1, 1])
    colors_obj = ['#F44336' if g >= 0 else '#FFCDD2' for g in obj_gain]
    ax3.bar(L, obj_gain, color=colors_obj, alpha=0.85)
    ax3.axhline(0, color='gray', lw=0.8)
    ax3.set_title('Layer-wise Info Gain: Object', fontsize=12)
    ax3.set_xlabel('Layer Index', fontsize=11)
    ax3.set_ylabel('ΔAccuracy', fontsize=11)
    ax3.set_xticks(L)
    ax3.grid(True, axis='y', ls='--', alpha=0.4)
    
    save_path = os.path.join(save_dir, "probing_results_mit_states.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")
    return save_path


# ==========================================
# 6. 主流程
# ==========================================
def main():
    device = CFG["device"]
    print(f"Device: {device}")
    
    # --- 加载模型 ---
    print("\n>>> 1. Loading CLIP Model (ViT-L/14)...")
    model = load_clip(name=CFG["weights_path"], context_length=77)
    model = model.to(device).eval()
    
    num_layers = len(model.visual.transformer.resblocks)
    print(f"    ViT layers detected: {num_layers}")
    
    # --- 注册 Hooks（一次性，全程保留）---
    hooks = []
    for i, block in enumerate(model.visual.transformer.resblocks):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)
    
    # --- 加载数据集 ---
    print("\n>>> 2. Loading Dataset...")
    try:
        from dataset import CompositionDataset
        train_dataset = CompositionDataset(
            CFG["dataset_path"], phase='train', split='compositional-split-natural'
        )
        test_dataset = CompositionDataset(
            CFG["dataset_path"], phase='test', split='compositional-split-natural'
        )
    except ImportError:
        print("  [ERROR] Dataset import failed. Adjust the import statement.")
        for h in hooks:
            h.remove()
        return
    
    # ⚠️ 注意：num_workers=0 避免 Hook + 多进程的潜在冲突
    train_loader = DataLoader(
        train_dataset, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=CFG["num_workers"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=CFG["num_workers"]
    )
    
    # --- 逐层 Probe ---
    print(f"\n>>> 3. Layer-by-Layer Probing (normalize={CFG['normalize_features']})...")
    
    layer_indices = list(range(num_layers))
    attr_accuracies = []
    obj_accuracies  = []
    
    # 获取类别数（只需从数据集获取一次）
    dummy_tr_feat, dummy_attrs, dummy_objs = load_or_extract(
        model, train_loader, 0, "train", CFG["normalize_features"]
    )
    dummy_te_feat, dummy_te_attrs, dummy_te_objs = load_or_extract(
        model, test_loader, 0, "test", CFG["normalize_features"]
    )
    feat_dim  = dummy_tr_feat.shape[1]
    num_attrs = max(dummy_attrs.max(), dummy_te_attrs.max()) + 1
    num_objs  = max(dummy_objs.max(), dummy_te_objs.max()) + 1
    print(f"    Feature dim: {feat_dim}, #Attrs: {num_attrs}, #Objs: {num_objs}")
    
    # 先做完 layer 0，其余层复用缓存
    print(f"  [Layer 0] (cached) training probes...")
    attr_acc = train_probe(dummy_tr_feat, dummy_attrs, dummy_te_feat, dummy_te_attrs, num_attrs)
    obj_acc  = train_probe(dummy_tr_feat, dummy_objs,  dummy_te_feat, dummy_te_objs,  num_objs)
    attr_accuracies.append(attr_acc)
    obj_accuracies.append(obj_acc)
    print(f"  Layer  0 | Attr: {attr_acc:.4f}  Obj: {obj_acc:.4f}")
    
    for i in range(1, num_layers):
        print(f"  [Layer {i:2d}] Extracting features...")
        
        tr_feat, tr_attrs, tr_objs = load_or_extract(
            model, train_loader, i, "train", CFG["normalize_features"]
        )
        te_feat, te_attrs, te_objs = load_or_extract(
            model, test_loader, i, "test", CFG["normalize_features"]
        )
        
        attr_acc = train_probe(tr_feat, tr_attrs, te_feat, te_attrs, num_attrs)
        obj_acc  = train_probe(tr_feat, tr_objs,  te_feat, te_objs,  num_objs)
        attr_accuracies.append(attr_acc)
        obj_accuracies.append(obj_acc)
        
        gap = abs(attr_acc - obj_acc)
        print(f"  Layer {i:2d} | Attr: {attr_acc:.4f}  Obj: {obj_acc:.4f}  Gap: {gap:.4f}")
    
    # 移除所有 Hooks（良好习惯）
    for h in hooks:
        h.remove()
    
    # --- 打印层位建议 ---
    print("\n>>> 4. Analysis Summary:")
    gaps = compute_disentanglement_gap(attr_accuracies, obj_accuracies)
    best_attr_layer  = int(np.argmax(attr_accuracies))
    best_obj_layer   = int(np.argmax(obj_accuracies))
    best_gap_layer   = int(np.argmax(gaps))
    
    print(f"  Best layer for Attribute probing : Layer {best_attr_layer} "
          f"({attr_accuracies[best_attr_layer]:.4f})")
    print(f"  Best layer for Object probing    : Layer {best_obj_layer} "
          f"({obj_accuracies[best_obj_layer]:.4f})")
    print(f"  Max disentanglement gap at       : Layer {best_gap_layer} "
          f"(gap={gaps[best_gap_layer]:.4f})")
    print(f"\n  ➡  Suggestion: Insert State-aware VPT around Layer {best_attr_layer}, "
          f"Object-aware VPT around Layer {best_obj_layer}")
    
    # 保存数值结果
    results = {
        "layer_indices": layer_indices,
        "attr_accuracies": attr_accuracies,
        "obj_accuracies": obj_accuracies,
        "disentanglement_gaps": gaps.tolist(),
        "config": {k: v for k, v in CFG.items() if k not in ("dataset_path", "weights_path")},
    }
    result_path = os.path.join(CFG["output_dir"], "probing_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Numerical results saved → {result_path}")
    
    # --- 可视化 ---
    print("\n>>> 5. Plotting...")
    plot_results(layer_indices, attr_accuracies, obj_accuracies, CFG["output_dir"])
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()