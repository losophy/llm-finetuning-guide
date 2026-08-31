# LoRA 秩和目标选择如何影响可训练参数数量
hidden_dim = 4096
intermediate_dim = 11008
num_layers = 32

def count_lora_params(rank, target="all"):
    attn_matrices = 4
    attn_params_per_layer = attn_matrices * 2 * (hidden_dim * rank)
    mlp_params_per_layer = (
        2 * (hidden_dim * rank + rank * intermediate_dim) +
        2 * (intermediate_dim * rank + rank * hidden_dim)
    )
    if target == "qv_only":
        params_per_layer = 2 * 2 * (hidden_dim * rank)
    elif target == "attention":
        params_per_layer = attn_params_per_layer
    else:
        params_per_layer = attn_params_per_layer + mlp_params_per_layer
    return params_per_layer * num_layers

print(f"{'秩':>6} | {'仅 QV':>12} | {'全部注意力':>12} | {'全部线性层':>12}")
print("-" * 55)
for rank in [4, 8, 16, 32, 64]:
    qv = count_lora_params(rank, "qv_only")
    attn = count_lora_params(rank, "attention")
    all_lin = count_lora_params(rank, "all")
    print(f"{rank:>6} | {qv/1e6:>10.2f}M | {attn/1e6:>10.2f}M | {all_lin/1e6:>10.2f}M")

total_rank16_all = count_lora_params(16, 'all')
print(f"\n7B 模型总参数: ~7,000M")
print(f"LoRA rank=16, 全部线性层: {total_rank16_all/1e6:.1f}M 可训练")
print(f"占总参数的 {total_rank16_all / 7e9 * 100:.2f}%")