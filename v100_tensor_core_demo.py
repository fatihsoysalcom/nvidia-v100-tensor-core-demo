import torch

# Check if CUDA is available and if the V100 GPU is detected
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")

    # Check if the GPU has Tensor Cores (V100 and later generally do)
    # This is a simplified check; actual Tensor Core usage depends on the operation.
    # PyTorch's matmul operations on supported dtypes (like float16, bfloat16) automatically leverage Tensor Cores.
    print("Assuming Tensor Cores are available and will be utilized by PyTorch for mixed-precision operations.")

    # Define matrix dimensions
    M, N, K = 1024, 1024, 1024

    # Create random matrices on the GPU
    # Using float16 (half precision) is crucial for Tensor Core acceleration
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)

    # Perform matrix multiplication (A @ B)
    # PyTorch automatically uses Tensor Cores for float16 matmul on compatible hardware.
    print(f"Performing matrix multiplication of {M}x{K} and {K}x{N} matrices using float16...")
    # Warm-up run
    for _ in range(10):
        C = A @ B

    # Time the operation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    # Actual timed operation
    C = A @ B
    end_time.record()

    # Synchronize to ensure all CUDA operations are complete
    torch.cuda.synchronize()

    elapsed_time_ms = start_time.elapsed_time(end_time)
    print(f"Matrix multiplication completed in {elapsed_time_ms:.2f} ms.")

    # Calculate theoretical FLOPs for matrix multiplication: 2 * M * N * K
    # For float16, Tensor Cores can perform 2x more operations per clock cycle than FP32 cores for certain operations.
    flops = 2 * M * N * K
    # Convert to TFLOPS (Tera Floating Point Operations Per Second)
    tflops = (flops / (elapsed_time_ms / 1000.0)) / 1e12
    print(f"Achieved performance: {tflops:.2f} TFLOPS (approximate).")

    # Example of a typical deep learning operation that benefits from Tensor Cores:
    # A convolutional layer operation can be expressed as a GEMM (General Matrix Multiply) operation.
    # PyTorch's nn.Conv2d with float16 inputs/weights on V100 will leverage Tensor Cores.

else:
    print("CUDA is not available. This demo requires a CUDA-enabled GPU, preferably an NVIDIA V100 or similar.")
    print("Please ensure you have a compatible GPU and the CUDA toolkit installed.")
