#!/bin/bash
#SBATCH --job-name=mine-mit             # 作业名称
#SBATCH --partition=gpu_mem             # 分区
#SBATCH --gres=gpu:1                    # 申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # MINE 无需大量 CPU（num_workers=0）
#SBATCH --time=06:00:00                 # 预估 1.5-2h，留足余量
#SBATCH --output=logs/mit-states/mine/%x-%j.out
#SBATCH --error=logs/mit-states/mine/%x-%j.err

# ─────────────────────────────────────────────
# 多层次视觉提示 - 互信息估计实验 (MINE) | MIT-States
# 依赖：已由 probe 实验生成的 feature_cache/*.npz
# ─────────────────────────────────────────────

# 1. 加载系统模块
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

# 3. 切换到工作目录（与 probing 实验保持一致，共享 feature_cache）
# cd /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Point2/code

# 4. 创建日志与输出目录
mkdir -p logs/mit-states/mine
mkdir -p mi_results

# 5. 检查特征缓存是否存在（前置依赖检查）
CACHE_DIR="./feature_cache"
REQUIRED_FILE="${CACHE_DIR}/layer0_train_normalized.npz"
if [ ! -f "${REQUIRED_FILE}" ]; then
    echo "[ERROR] 特征缓存不存在: ${REQUIRED_FILE}"
    echo "        请先运行 probing 实验（run_probe.sh）生成缓存后再执行本脚本。"
    exit 1
fi

CACHE_COUNT=$(ls ${CACHE_DIR}/layer*_normalized.npz 2>/dev/null | wc -l)
echo ">>> 检测到特征缓存文件: ${CACHE_COUNT} 个"

# 6. 打印 GPU 状态
echo -e "\n>>> 当前 GPU 硬件分配状态 <<<"
nvidia-smi

# 7. 打印实验信息
echo "========================================"
echo "  MINE 互信息估计实验  |  MIT-States"
echo "  SLURM Job ID : ${SLURM_JOB_ID:-manual}"
echo "  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Working Dir  : $(pwd)"
echo "  Cache Dir    : ${CACHE_DIR} (${CACHE_COUNT} files)"
echo "  Output Dir   : ./mi_results"
echo "  Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 8. 运行 MINE 实验 sbatch probe_exp/scripts/run_mine.sh   
python -u probe_exp/mine.py

EXIT_CODE=$?

# 9. 结束信息
echo "========================================"
echo "  End Time : $(date '+%Y-%m-%d %H:%M:%S')"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  状态     : ✅ 实验完成"
    echo ""
    echo "  生成文件："
    echo "    mi_results/mi_estimation_results.png   # 主可视化图"
    echo "    mi_results/mi_results.json             # 各层 MI 数值"
    echo "    mi_results/mi_vs_linear_probe.json     # 与 Linear Probe 对比"
    echo ""
    echo "  关键结论（从 JSON 读取）："
    python -c "
import json, os
p = './mi_results/mi_results.json'
if os.path.exists(p):
    d = json.load(open(p))
    mi_a = d['mi_attr']
    mi_o = d['mi_obj']
    gap  = d['mi_gap']
    print(f'    Max MI (Attr): Layer {mi_a.index(max(mi_a)):2d}  ->  {max(mi_a):.4f} nats')
    print(f'    Max MI (Obj) : Layer {mi_o.index(max(mi_o)):2d}  ->  {max(mi_o):.4f} nats')
    print(f'    Max MI Gap   : Layer {gap.index(max(gap)):2d}  ->  {max(gap):.4f} nats')
"
else
    echo "  状态     : ❌ 实验异常退出 (exit code: ${EXIT_CODE})"
    echo "  请检查错误日志: logs/mit-states/mine/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
fi
echo "========================================"

exit ${EXIT_CODE}