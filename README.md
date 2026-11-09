# NVIDIA V100 Tensor Core Demo

This Python script demonstrates how to leverage NVIDIA V100 Tensor Cores for accelerated matrix multiplication using PyTorch. It performs a large matrix multiplication in float16 precision on the GPU and measures the execution time to estimate performance in TFLOPS.

## Language

`python`

## How to Run

1. Ensure you have an NVIDIA GPU with CUDA support (preferably V100 or newer).
2. Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (adjust cuXXX for your CUDA version).
3. Run the script: `python v100_tensor_core_demo.py`

## Original Article

This example accompanies the Turkish article: [NVIDIA V100 Tensor Çekirdeklerinden Maksimum Verim Alma](https://fatihsoysal.com/blog/nvidia-v100-tensor-cekirdeklerinden-maksimum-verim-alma/).

## License

MIT — see [LICENSE](LICENSE).
