# 计算全参数微调 vs LoRA 所需的 GPU 内存
# 纯算术——运行此单元不需要 GPU

model_params = 7_000_000_000  # 70 亿参数

bytes_per_param_bf16 = 2  # 16-bit = 2 字节
bytes_per_param_fp32 = 4  # 32-bit = 4 字节

# 全参数微调：权重 + 梯度 + Adam 优化器状态
# Adam 会存储动量 + 方差 = 每个梯度的 2 份额外 fp32 副本
weights_gb = model_params * bytes_per_param_bf16 / 1e9
gradients_gb = model_params * bytes_per_param_fp32 / 1e9
optimizer_gb = model_params * bytes_per_param_fp32 * 2 / 1e9
total_full_ft_gb = weights_gb + gradients_gb + optimizer_gb

print("=== 全参数微调 7B 模型所需内存 ===")
print(f"  权重 (bf16):         {weights_gb:.1f} GB")
print(f"  梯度 (fp32):       {gradients_gb:.1f} GB")
print(f"  Adam 优化器状态:  {optimizer_gb:.1f} GB")
print(f"  总计:                  {total_full_ft_gb:.1f} GB")

# LoRA 微调：只有 adapter 矩阵更新
# 32 个 Transformer 层 × 4 个注意力矩阵 = 128 个 adapter 对
lora_rank = 16
num_adapter_pairs = 32 * 4
hidden_dim = 4096

lora_params = num_adapter_pairs * 2 * (hidden_dim * lora_rank)

lora_weights_gb = lora_params * bytes_per_param_bf16 / 1e9
lora_grads_gb = lora_params * bytes_per_param_fp32 / 1e9
lora_optim_gb = lora_params * bytes_per_param_fp32 * 2 / 1e9
base_model_gb = model_params * bytes_per_param_bf16 / 1e9
total_lora_gb = base_model_gb + lora_weights_gb + lora_grads_gb + lora_optim_gb

print(f"\n=== LoRA 微调所需内存 (rank={lora_rank}, 仅注意力层) ===")
print(f"  冻结的基础模型 (bf16):  {base_model_gb:.1f} GB")
print(f"  LoRA 可训练参数:     {lora_params:,}")
print(f"  LoRA 权重 + 梯度 + 优化器: {(lora_weights_gb + lora_grads_gb + lora_optim_gb):.2f} GB")
print(f"  总计:                     {total_lora_gb:.1f} GB")
print(f"  相比全参数微调的内存缩减:      {total_full_ft_gb / total_lora_gb:.1f}x")
