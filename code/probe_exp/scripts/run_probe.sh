#!/bin/bash
#SBATCH --job-name=probe-mit            # 作业名称
#SBATCH --partition=gpu_mem             # 分区
#SBATCH --gres=gpu:1                    # 申请 1 块 GPU (特征提取需要GPU加速)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # CPU 核数 (用于 DataLoader 多线程读取)
#SBATCH --time=24:00:00                 # 运行时间限制
#SBATCH --output=logs/mit-states/probe/%x-%j.out  # 标准输出日志路径
#SBATCH --error=logs/mit-states/probe/%x-%j.err   # 错误输出日志路径

# ─────────────────────────────────────────────
# 多层次视觉提示 - 线性探测实验 (Probing) | MIT-States
# ─────────────────────────────────────────────

# 1. 加载系统模块
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

# 4. 创建日志目录
mkdir -p logs/mit-states/probe

# [新增] 打印当前 GPU 资源与状态信息
echo -e "\n>>> 当前 GPU 硬件分配状态 <<<"
nvidia-smi

# 5. 打印环境信息
echo "========================================"
echo "  Probing Experiment  |  MIT-States"
echo "  SLURM Job ID : ${SLURM_JOB_ID:-manual}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Working Dir  : $(pwd)"
echo "========================================"

# 6. 运行训练（-u 实时输出日志，避免缓冲导致在 .out 文件中看不到进度条）
python -u probe_exp/probing_experiment.py

echo "========================================"
echo "  Probing 实验完成！"
echo "  请在工作目录下检查生成的图片文件："
echo "  probing_results_mit_states.png"
echo "========================================"