import ntl
import build.cipher_cpp as cip_cpp
import torch
import singleton_timer as st
import numpy as np
import random
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import math
import ntl

wt_matmul = None
wt_matmul_s8x_s8y_s32z = None


def UnsecureMatmul(x: torch.Tensor, y: torch.Tensor):
    timer = st.SingletonTimer()

    global wt_matmul_s8x_s8y_s32z
    if wt_matmul_s8x_s8y_s32z is None:
        with open('./cipher_cuda/warptile_matmul_s8x_s8y_s32z.cuh', 'r') as fin:
            cuda_kernel = fin.read()
        mod = SourceModule(cuda_kernel)
        wt_matmul_s8x_s8y_s32z = mod.get_function("sgemmWarptiling")

    assert x.size()[1] == y.size()[0]
    M, K = x.size()
    _, N = y.size()

    # Type must be int32 to properly handle negatives
    t = timer.start(tag='torch to numpy, Unsecure',
                    category='torch to numpy, Unsecure')
    x_np = x.numpy().astype(np.int8)
    y_np = y.numpy().astype(np.int8)
    timer.end(t)

    t = timer.start(tag='Allocate z_np (CPU), Unsecure',
                    category='Allocate z_np (CPU), Unsecure')
    z_np = np.zeros((M, N), dtype=np.int32)
    timer.end(t)

    t = timer.start(tag='Allocate x_np, y_np, z_np (GPU), Unsecure',
                    category='Allocate x_np, y_np, z_np (GPU), Unsecure')
    x_np_gpu = cuda.mem_alloc(x_np.nbytes)
    y_np_gpu = cuda.mem_alloc(y_np.nbytes)
    z_np_gpu = cuda.mem_alloc(z_np.nbytes)
    timer.end(t)

    t = timer.start(tag='Transfer x_np, y_np from CPU to GPU, Unsecure',
                    category='Transfer x_np, y_np from CPU to GPU, Unsecure')
    cuda.memcpy_htod(x_np_gpu, x_np)
    cuda.memcpy_htod(y_np_gpu, y_np)
    timer.end(t)

    t = timer.start(tag='GPU Computation (Pre), Unsecure',
                    category='GPU Computation (Pre), Unsecure')
    NUM_THREADS = 128
    BN = 128
    BM = 128

    block_size = (NUM_THREADS, 1, 1)
    grid_size = (
        int(np.ceil(N / BN)), int(np.ceil(M / BM)))
    timer.end(t)

    t = timer.start(tag='GPU Computation, Unsecure',
                    category='GPU Computation, Unsecure')
    wt_matmul_s8x_s8y_s32z(x_np_gpu, y_np_gpu, z_np_gpu, np.int32(M), np.int32(N),
                           np.int32(K), block=block_size, grid=grid_size)
    cuda.Context.synchronize()
    timer.end(t)

    t = timer.start(tag='Transfer z_np from GPU to CPU, Unsecure',
                    category='Transfer z_np from GPU to CPU, Unsecure')
    cuda.memcpy_dtoh(z_np, z_np_gpu)
    timer.end(t)

    t = timer.start(tag='numpy to torch, Unsecure',
                    category='numpy to torch, Unsecure')
    z = torch.from_numpy(z_np)
    timer.end(t)
    return z


def SecureMatmul(x: torch.Tensor, y: torch.Tensor):
    assert x.dtype == torch.int8 and y.dtype == torch.int8
    if x.dtype == torch.int8:
        B = 8
    timer = st.SingletonTimer()

    global wt_matmul
    if wt_matmul is None:
        with open('./cipher_cuda/warptile_matmul.cuh', 'r') as fin:
            cuda_kernel = fin.read()
        mod = SourceModule(cuda_kernel)
        wt_matmul = mod.get_function("sgemmWarptiling")

    assert x.size()[1] == y.size()[0]
    M, K = x.size()
    _, N = y.size()

    mod = 2 ** 32

    t = timer.start(tag='Key Gen', category='Key Gen')

    # a1, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    # b1 = random.randint(0, mod)
    # a2, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    # b2 = random.randint(0, mod)

    key_set_a1 = ntl.ntl_cpp.GenerateKeySetA(mod, M)
    key_set_b1 = ntl.ntl_cpp.GenerateKeySetB(mod, M)
    key_set_a2 = ntl.ntl_cpp.GenerateKeySetA(mod, N)
    key_set_b2 = ntl.ntl_cpp.GenerateKeySetB(mod, N)

    timer.end(t)
    # print(f'a1 = {a1}, b1 = {b1}, a2 = {a2}, b2 = {b2}')

    # Type must be int32 to properly handle negatives
    t = timer.start(tag='torch to numpy', category='torch to numpy')
    x_np = x.numpy().astype(np.int32)
    y_np = y.numpy().astype(np.int32)
    timer.end(t)

    # Compute Row sum and Col sum for shifting, Need copy to be hidden
    t = timer.start(tag='Shift SumByRow Metadata Gen',
                    category='Shift SumByRow Metadata Gen')
    shift_row_sum_x = cip_cpp.SumByRow_int32(x_np)
    timer.end(t)
    t = timer.start(tag='Shift SumByCol Metadata Gen',
                    category='Shift SumByCol Metadata Gen')
    shift_col_sum_y = cip_cpp.SumByCol_int32(y_np)
    timer.end(t)

    # Now need to shift
    t = timer.start(tag='Shift', category='Shift')
    amt = 2**(B-1)
    cip_cpp.Shift_int32(x_np, amt)
    cip_cpp.Shift_int32(y_np, amt)
    timer.end(t)

    # Compute Row sum and Col sum for Decryption (can be hidden)
    t = timer.start(tag='Dec SumByRow Metadata Gen',
                    category='Dec SumByRow Metadata Gen')
    dec_row_sum_x = cip_cpp.SumByRow_int32(x_np)
    timer.end(t)

    t = timer.start(tag='Dec SumByCol Metadata Gen',
                    category='Dec SumByCol Metadata Gen')
    dec_col_sum_y = cip_cpp.SumByCol_int32(y_np)
    timer.end(t)

    # Need to encrypt
    t = timer.start(tag='int32 to uint32', category='int32 to uint32')
    x_np = x_np.astype(np.uint32)
    y_np = y_np.astype(np.uint32)
    timer.end(t)

    t = timer.start(tag='Encryption', category='Encryption')
    cip_cpp.EncryptMatrix2D(x_np, key_set_a1, key_set_b1, mod, False)
    cip_cpp.EncryptMatrix2D(y_np, key_set_a2, key_set_b2, mod, True)
    timer.end(t)

    t = timer.start(tag='Allocate z_np (CPU)', category='Allocate z_np (CPU)')
    z_np = np.zeros((M, N), dtype=np.uint32)
    timer.end(t)

    t = timer.start(tag='Allocate x_np, y_np, z_np (GPU)',
                    category='Allocate x_np, y_np, z_np (GPU)')
    x_np_gpu = cuda.mem_alloc(x_np.nbytes)
    y_np_gpu = cuda.mem_alloc(y_np.nbytes)
    z_np_gpu = cuda.mem_alloc(z_np.nbytes)
    timer.end(t)

    t = timer.start(tag='Transfer x_np, y_np from CPU to GPU',
                    category='Transfer x_np, y_np from CPU to GPU')
    cuda.memcpy_htod(x_np_gpu, x_np)
    cuda.memcpy_htod(y_np_gpu, y_np)
    timer.end(t)

    t = timer.start(tag='GPU Computation (Pre)',
                    category='GPU Computation (Pre)')
    NUM_THREADS = 128
    BN = 128
    BM = 128

    block_size = (NUM_THREADS, 1, 1)
    grid_size = (
        int(np.ceil(N / BN)), int(np.ceil(M / BM)))
    timer.end(t)

    t = timer.start(tag='GPU Computation', category='GPU Computation')
    wt_matmul(x_np_gpu, y_np_gpu, z_np_gpu, np.int32(M), np.int32(N),
              np.int32(K), block=block_size, grid=grid_size)
    cuda.Context.synchronize()
    timer.end(t)

    t = timer.start(tag='Transfer z_np from GPU to CPU',
                    category='Transfer z_np from GPU to CPU')
    cuda.memcpy_dtoh(z_np, z_np_gpu)
    timer.end(t)

    # t = timer.start(tag='Find Key Inv Mod',
    #                 category='Find Key Inv Mod')
    # # Compute Key inv mod
    # a_12_key_inv = ntl.FindKeyInvMod(a1 * a2, mod)
    # timer.end(t)

    t = timer.start(tag='Decryption',
                    category='Decryption')
    cip_cpp.DecryptMatrix2D(z_np, K, key_set_a1, key_set_b1, key_set_a2, key_set_b2,
                            mod, dec_row_sum_x, dec_col_sum_y)
    timer.end(t)

    t = timer.start(tag='uint32 to int32', category='uint32 to int32')
    z_np = z_np.astype(np.int32)
    timer.end(t)

    t = timer.start(tag='UndoShift', category='UndoShift')
    cip_cpp.UndoShift_int32(z_np, amt, K, shift_row_sum_x, shift_col_sum_y)
    timer.end(t)

    t = timer.start(tag='numpy to torch', category='numpy to torch')
    z = torch.from_numpy(z_np)
    timer.end(t)
    return z

def SecureMatmulFullLevel(x: torch.Tensor, y: torch.Tensor):
    assert x.dtype == torch.int8 and y.dtype == torch.int8
    if x.dtype == torch.int8:
        B = 8
    timer = st.SingletonTimer()
    assert B <= 8

    global wt_matmul
    if wt_matmul is None:
        with open('./cipher_cuda/warptile_matmul.cuh', 'r') as fin:
            cuda_kernel = fin.read()
        mod = SourceModule(cuda_kernel)
        wt_matmul = mod.get_function("sgemmWarptiling")

    assert x.size()[1] == y.size()[0]
    M, K = x.size()
    _, N = y.size()

    mod = 2 ** 32

    t = timer.start(tag='Key Gen', category='Key Gen')

    # a1, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    # b1 = random.randint(0, mod)
    # a2, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    # b2 = random.randint(0, mod)

    key_set_a1 = ntl.ntl_cpp.GenerateKeySetA(mod, M)
    key_set_b1 = ntl.ntl_cpp.GenerateKeySetB(mod, M)
    key_set_a2 = ntl.ntl_cpp.GenerateKeySetA(mod, N)
    key_set_b2 = ntl.ntl_cpp.GenerateKeySetB(mod, N)

    timer.end(t)
    # print(f'a1 = {a1}, b1 = {b1}, a2 = {a2}, b2 = {b2}')

    # Type must be int32 to properly handle negatives
    t = timer.start(tag='torch to numpy', category='torch to numpy')
    x_np = x.numpy().astype(np.int32)
    y_np = y.numpy().astype(np.int32)
    timer.end(t)

    # Compute Row sum and Col sum for shifting, Need copy to be hidden
    t = timer.start(tag='Shift SumByRow Metadata Gen',
                    category='Shift SumByRow Metadata Gen')
    shift_row_sum_x = cip_cpp.SumByRow_int32(x_np)
    timer.end(t)
    t = timer.start(tag='Shift SumByCol Metadata Gen',
                    category='Shift SumByCol Metadata Gen')
    shift_col_sum_y = cip_cpp.SumByCol_int32(y_np)
    timer.end(t)

    # Now need to shift
    t = timer.start(tag='Shift', category='Shift')
    amt = 2**(B-1)
    cip_cpp.Shift_int32(x_np, amt)
    cip_cpp.Shift_int32(y_np, amt)
    timer.end(t)

    # Compute Row sum and Col sum for Decryption (can be hidden)
    t = timer.start(tag='Dec SumByRow Metadata Gen',
                    category='Dec SumByRow Metadata Gen')
    dec_row_sum_x = cip_cpp.SumByRow_int32(x_np)
    timer.end(t)

    t = timer.start(tag='Dec SumByCol Metadata Gen',
                    category='Dec SumByCol Metadata Gen')
    dec_col_sum_y = cip_cpp.SumByCol_int32(y_np)
    timer.end(t)

    # Need to encrypt
    t = timer.start(tag='int32 to uint32', category='int32 to uint32')
    x_np = x_np.astype(np.uint32)
    y_np = y_np.astype(np.uint32)
    timer.end(t)

    t = timer.start(tag='Encryption', category='Encryption')
    cip_cpp.EncryptMatrix2D(x_np, key_set_a1, key_set_b1, mod, False)
    cip_cpp.EncryptMatrix2D(y_np, key_set_a2, key_set_b2, mod, True)
    timer.end(t)

    t = timer.start(tag='Allocate z_np (CPU)', category='Allocate z_np (CPU)')
    z_np = np.zeros((M, N), dtype=np.uint32)
    timer.end(t)

    t = timer.start(tag='Allocate x_np, y_np, z_np (GPU)',
                    category='Allocate x_np, y_np, z_np (GPU)')
    x_np_gpu = cuda.mem_alloc(x_np.nbytes)
    y_np_gpu = cuda.mem_alloc(y_np.nbytes)
    z_np_gpu = cuda.mem_alloc(z_np.nbytes)
    timer.end(t)

    t = timer.start(tag='Transfer x_np, y_np from CPU to GPU',
                    category='Transfer x_np, y_np from CPU to GPU')
    cuda.memcpy_htod(x_np_gpu, x_np)
    cuda.memcpy_htod(y_np_gpu, y_np)
    timer.end(t)

    t = timer.start(tag='GPU Computation (Pre)',
                    category='GPU Computation (Pre)')
    NUM_THREADS = 128
    BN = 128
    BM = 128

    block_size = (NUM_THREADS, 1, 1)
    grid_size = (
        int(np.ceil(N / BN)), int(np.ceil(M / BM)))
    timer.end(t)

    t = timer.start(tag='GPU Computation', category='GPU Computation')
    wt_matmul(x_np_gpu, y_np_gpu, z_np_gpu, np.int32(M), np.int32(N),
              np.int32(K), block=block_size, grid=grid_size)
    cuda.Context.synchronize()
    timer.end(t)

    t = timer.start(tag='Transfer z_np from GPU to CPU',
                    category='Transfer z_np from GPU to CPU')
    cuda.memcpy_dtoh(z_np, z_np_gpu)
    timer.end(t)

    # t = timer.start(tag='Find Key Inv Mod',
    #                 category='Find Key Inv Mod')
    # # Compute Key inv mod
    # a_12_key_inv = ntl.FindKeyInvMod(a1 * a2, mod)
    # timer.end(t)

    t = timer.start(tag='Decryption',
                    category='Decryption')
    cip_cpp.DecryptMatrix2D(z_np, K, key_set_a1, key_set_b1, key_set_a2, key_set_b2,
                            mod, dec_row_sum_x, dec_col_sum_y)
    timer.end(t)

    t = timer.start(tag='uint32 to int32', category='uint32 to int32')
    z_np = z_np.astype(np.int32)
    timer.end(t)

    t = timer.start(tag='UndoShift', category='UndoShift')
    cip_cpp.UndoShift_int32(z_np, amt, K, shift_row_sum_x, shift_col_sum_y)
    timer.end(t)

    t = timer.start(tag='numpy to torch', category='numpy to torch')
    z = torch.from_numpy(z_np)
    timer.end(t)
    return z

