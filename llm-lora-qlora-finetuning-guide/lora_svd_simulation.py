import numpy as np

# 用 numpy 模拟 LoRA 的低秩分解
np.random.seed(42)

d_out, d_in = 512, 512

# 全参数微调会产生的权重更新
delta_W_full = np.random.randn(d_out, d_in).astype(np.float32) * 0.01

# LoRA：用两个小矩阵 A 和 B 近似 delta_W
# B 初始化为 ZERO——模型一开始与预训练基础模型完全相同
rank = 16
A = np.random.randn(rank, d_in).astype(np.float32) * 0.01
B = np.zeros((d_out, rank), dtype=np.float32)

# 用 SVD 模拟训练后的 adapter——只保留前 'rank' 个奇异向量
# U 和 Vt 捕获最重要的方向；S 包含它们的幅度
U, S, Vt = np.linalg.svd(delta_W_full, full_matrices=False)
A_trained = np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]
B_trained = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
delta_W_lora = B_trained @ A_trained

print(f"原始权重矩阵: {d_out} × {d_in} = {d_out * d_in:,} 个数值")
print(f"\nLoRA 分解 (rank={rank}):")
print(f"  矩阵 A: {A_trained.shape}  →  {A_trained.size:,} 个参数")
print(f"  矩阵 B: {B_trained.shape}  →  {B_trained.size:,} 个参数")
print(f"  LoRA 总参数: {A_trained.size + B_trained.size:,}")
print(f"  全量更新:    {delta_W_full.size:,}")
print(f"  参数缩减: {delta_W_full.size / (A_trained.size + B_trained.size):.0f}x")