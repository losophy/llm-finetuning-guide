# 练习 1：统计 Llama 2 13B 的 LoRA 参数
hidden_dim_13b = 5120
intermediate_dim_13b = 13824
num_layers_13b = 40
rank = 16

def count_lora_params_13b(rank):
    attn_params = 4 * 2 * (hidden_dim_13b * rank)
    mlp_params = (
        2 * (hidden_dim_13b * rank + rank * intermediate_dim_13b) +
        2 * (intermediate_dim_13b * rank + rank * hidden_dim_13b)
    )
    return (attn_params + mlp_params) * num_layers_13b

total_lora = count_lora_params_13b(rank)
total_model = 13_000_000_000
print(f"LoRA 可训练参数: {total_lora / 1e6:.1f}M")
print(f"LoRA 占比: {total_lora / total_model * 100:.2f}%")

# 答案：hidden_dim_13b=5120, intermediate_dim_13b=13824, num_layers_13b=40
# 预期：约 78.6M 可训练参数，占 13B 的 0.60%