"""GGUF 导出脚本 - 用于 Ollama/llama.cpp 部署"""


def export_to_gguf(model, tokenizer, output_dir="./qwen2.5-gguf",
                   quantization_method="q4_k_m"):
    """导出模型为 GGUF 格式"""

    # 支持的量化方法（7B 体积为参考值）
    quant_methods = {
        "q4_k_m": "4-bit，≈4.4GB，质量与体积平衡（Unsloth 默认，本项目选用）",
        "q5_k_m": "5-bit，≈5.4GB，质量更高",
        "q6_k": "6-bit，≈6.6GB，较高质量",
        "q8_0": "8-bit，≈8GB，接近原精度",
        "q3_k_m": "3-bit，≈3.5GB，更省空间，质量有损",
    }

    if quantization_method not in quant_methods:
        print(f"不支持的量化方法: {quantization_method}")
        print(f"支持的方法: {list(quant_methods.keys())}")
        return

    print(f"正在导出为 GGUF ({quantization_method})...")

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quantization_method
    )

    print(f"GGUF 模型已保存到: {output_dir}")
    print(f"量化方法: {quant_methods[quantization_method]}")


if __name__ == "__main__":
    # 示例用法
    # export_to_gguf(model, tokenizer)
    # export_to_gguf(model, tokenizer, quantization_method="q4_k_m")

    print("GGUF 导出脚本")
    print("支持的量化方法（7B 体积为参考值）:")
    print("  q4_k_m - 4-bit，≈4.4GB，质量与体积平衡（本项目选用）")
    print("  q5_k_m - 5-bit，≈5.4GB，质量更高")
    print("  q6_k - 6-bit，≈6.6GB，较高质量")
    print("  q8_0 - 8-bit，≈8GB，接近原精度")
    print("  q3_k_m - 3-bit，≈3.5GB，更省空间，质量有损")
