"""GGUF 导出脚本 - 用于 Ollama/llama.cpp 部署"""


def export_to_gguf(model, tokenizer, output_dir="./qwen2.5-gguf",
                   quantization_method="q4_k_m"):
    """导出模型为 GGUF 格式"""

    # 支持的量化方法
    quant_methods = {
        "q4_k_m": "4-bit，推荐（默认）",
        "q8_0": "8-bit，质量更高",
        "q5_k_m": "5-bit，平衡选择",
        "q6_k": "6-bit，较高质量",
        "q3_k_m": "3-bit，最小体积",
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
    # export_to_gguf(model, tokenizer, quantization_method="q8_0")

    print("GGUF 导出脚本")
    print("支持的量化方法:")
    print("  q4_k_m - 4-bit，推荐")
    print("  q8_0 - 8-bit，质量更高")
    print("  q5_k_m - 5-bit，平衡选择")
