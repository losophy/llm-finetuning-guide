"""验证 Unsloth 安装环境"""

import torch


def verify_installation():
    """检查 PyTorch、CUDA 和 GPU 信息"""
    print("=" * 50)
    print("Unsloth 安装验证")
    print("=" * 50)

    # PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")

    # CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")

    if cuda_available:
        # GPU 信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_mem / 1e9

        print(f"GPU 名称: {gpu_name}")
        print(f"VRAM 容量: {vram_gb:.1f} GB")
        print(f"计算能力: {gpu_props.major}.{gpu_props.minor}")

        # BF16 支持
        bf16_support = torch.cuda.is_bf16_supported()
        print(f"BF16 支持: {bf16_support}")
    else:
        print("警告: 未检测到 NVIDIA GPU")
        print("请确保已安装 CUDA 驱动")

    print("=" * 50)

    return cuda_available


if __name__ == "__main__":
    verify_installation()
