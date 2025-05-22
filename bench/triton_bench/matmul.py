import torch
import triton
import triton.language as tl
import time
import numpy as np

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

# Helper function to compute pid
@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

# Basic matmul kernel
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        stride_am, stride_ak, 
        stride_bk, stride_bn, 
        stride_cm, stride_cn,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, 
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Basic TMA matmul kernel
@triton.jit
def matmul_tma_ws_kernel(
        a_ptr, b_ptr, c_ptr,
        a_stride0, a_stride1,
        b_stride0, b_stride1,
        c_stride0, c_stride1,
        M, N, K,
        num_stages: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        USE_FP8: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid(pid, num_pid_n, num_pid_m, GROUP_SIZE_M)

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    off_am = pid_m * BLOCK_SIZE_M
    off_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
        off_k = k * BLOCK_SIZE_K
        a = a_desc.load((off_am, off_k))
        b = b_desc.load((off_bn, off_k))
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float8e4nv if USE_FP8 else tl.float16)
    c_desc.store((off_am, off_bn), c)

# Persistent TMA matmul kernel
@triton.jit
def matmul_tma_persistent_ws_kernel(
        a_ptr, b_ptr, c_ptr,
        a_stride0, a_stride1,
        b_stride0, b_stride1,
        c_stride0, c_stride1,
        M, N, K,
        num_stages: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
        USE_FP8: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                     block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                     block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                     block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=True, num_stages=num_stages):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)

        off_am = pid_m * BLOCK_SIZE_M
        off_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            off_k = ki * BLOCK_SIZE_K
            a = a_desc.load((off_am, off_k))
            b = b_desc.load((off_bn, off_k))
            accumulator = tl.dot(a, b.T, accumulator)

        c = accumulator.to(tl.float8e4nv if USE_FP8 else tl.float16)
        c_desc.store((off_am, off_bn), c)

def _matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 64,
    num_stages: int = 3,
    num_warps: int = 8,
    use_fp8: bool = False,
    kernel_type: str = "tma",
    num_sms: int = None
):
    if num_sms is None and torch.cuda.is_available():
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    
    if exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, use_fp8):
        raise ValueError(f"配置 {BLOCK_SIZE_M}x{BLOCK_SIZE_N}x{BLOCK_SIZE_K} 超出共享内存限制")

    triton.set_allocator(lambda size, align, stream: 
        torch.empty(size, dtype=torch.int8, device="cuda"))

    if kernel_type == "basic":
        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
        matmul_kernel[grid](
            a_ptr, b_ptr, c_ptr,
            a_ptr.stride(0), a_ptr.stride(1),
            b_ptr.stride(0), b_ptr.stride(1),
            c_ptr.stride(0), c_ptr.stride(1),
            M, N, K,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            GROUP_SIZE_M=8,
        )
    elif kernel_type == "tma":
        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
        matmul_tma_ws_kernel[grid](
            a_ptr, b_ptr, c_ptr,
            a_ptr.stride(0), a_ptr.stride(1),
            b_ptr.stride(0), b_ptr.stride(1),
            c_ptr.stride(0), c_ptr.stride(1),
            M, N, K, num_stages,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            GROUP_SIZE_M=8, USE_FP8=use_fp8,
            num_warps=num_warps
        )
    elif kernel_type == "persistent":
        grid = (min(num_sms, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),)
        matmul_tma_persistent_ws_kernel[grid](
            a_ptr, b_ptr, c_ptr,
            a_ptr.stride(0), a_ptr.stride(1),
            b_ptr.stride(0), b_ptr.stride(1),
            c_ptr.stride(0), c_ptr.stride(1),
            M, N, K, num_stages,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            GROUP_SIZE_M=8, NUM_SMS=num_sms,
            USE_FP8=use_fp8, num_warps=num_warps
        )
    else:
        raise ValueError(f"Invalid kernel_type: {kernel_type}")
    return c_ptr


def exceeds_smem_capacity(num_stages, BLOCK_M, BLOCK_N, BLOCK_K, use_fp8):
    return (num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) + BLOCK_M * BLOCK_N) * (1 if use_fp8 else 2) > 228 * 1024

def benchmark_matmul(M, N, K, kernel_type='tma'):
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps, use_fp8
    configs = [
        (128, 128, 64, 3, 8, False),
        (128, 256, 64, 3, 8, False),
        (128, 128, 128, 4, 8, False),
        # (128, 128, 64, 3, 8, True),
        # (128, 256, 64, 3, 8, True),
        # (128, 128, 128, 4, 8, True),
    ]
    for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps, use_fp8 in configs:
        # 共享内存检查
        if exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, use_fp8):
            print(f"Skipping config {BLOCK_SIZE_M}x{BLOCK_SIZE_N}x{BLOCK_SIZE_K} (exceeds shared memory)")
            continue
        
        dtype = torch.float8_e4m3fn if use_fp8 else torch.float16
        device = "cuda"
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)
        
        if kernel_type == 'cublas':
            for _ in range(5):
                cublas.matmul(A, B, C)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                cublas.matmul(A, B, C)
            torch.cuda.synchronize()
            duration = (time.time() - start) / 100
            ms = duration * 1e3
        elif kernel_type == 'torch':
            for _ in range(5):
                torch.matmul(A, B)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                torch.matmul(A, B)
            torch.cuda.synchronize()
            duration = (time.time() - start) / 100
            ms = duration * 1e3
        else:
            # 设置内存分配器
            triton.set_allocator(lambda size, align, stream: 
                torch.empty(size, dtype=torch.int8, device="cuda"))

            fn = lambda: _matmul(A, B, C, M, N, K, kernel_type=kernel_type)
            C = fn()
            ms = triton.testing.do_bench(fn)
            

        flops = 2 * M * N * K  
        # tflops = flops / (duration * 1e12)
        tflops = flops * 1e-12 / (ms * 1e-3)
        # ref_out = torch.matmul(A, B)
        # print(ref_out)
        # print(C)
        # torch.testing.assert_close(ref_out, C, atol=0.1, rtol=0.1)
        # ref_out = torch.empty_like(C)
        # cublas.matmul(A, B, ref_out)
        # torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)
        return {
            'duration_ms': ms,
            'tflops': tflops,
            'kernel_type': kernel_type,
            'config': {
                'M': M, 'N': N, 'K': K,
                'BLOCK_SIZE_M': BLOCK_SIZE_M,
                'BLOCK_SIZE_N': BLOCK_SIZE_N,
                'BLOCK_SIZE_K': BLOCK_SIZE_K,
                'num_stages': num_stages,
                'num_warps': num_warps,
                'use_fp8': use_fp8
            }
        }

def main():
    sizes = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    print("Running TMA MatMul Benchmarks...")
    print("-" * 70)
    print(f"{'Size (M,N,K)':>19} | {'Kernel Type':>15} | {'Time (ms)':>10} | {'TFLOPS':>10}")
    print("-" * 70)
    
    for M, N, K in sizes:
        for kernel_type in ['torch', 'cublas', 'basic', 'tma', 'persistent']:
            try:
                results = benchmark_matmul(M, N, K, kernel_type)
                print(f"({M:>5},{N:>5},{K:>5}) | {kernel_type:>15} | {results['duration_ms']:>10.2f} | {results['tflops']:>10.2f}")
            except Exception as e:
                print(f"({M:>5},{N:>5},{K:>5}) | {kernel_type:>15} | Failed: {str(e)}")
        print("-" * 70)

if __name__ == "__main__":
    main()