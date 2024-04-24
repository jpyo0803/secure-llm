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
import cupy as cp
wt_matmul = None
wt_matmul_s8x_s8y_s32z = None

if wt_matmul is None:
    with open('./cipher_cuda/warptile_matmul.cuh', 'r') as fin:
        cuda_kernel = fin.read()
    mod = SourceModule(cuda_kernel)
    wt_matmul = mod.get_function("sgemmWarptiling")

if wt_matmul_s8x_s8y_s32z is None:
    with open('./cipher_cuda/warptile_matmul_s8x_s8y_s32z.cuh', 'r') as fin:
        cuda_kernel = fin.read()
    mod = SourceModule(cuda_kernel)
    wt_matmul_s8x_s8y_s32z = mod.get_function("sgemmWarptiling")

print("cipher lib initilzied!")


def BatchUnsecureMatmul(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 3 and y.ndim == 3, "Input tensors must be 3D (batched matrices)"
    assert x.shape[0] == y.shape[0], "Number of batches must be equal"
    assert x.dtype == np.int8 and y.dtype == np.int8

    batch_size, M, K = x.shape
    _, K, N = y.shape

    results = []

    for i in range(batch_size):
        results.append(UnsecureMatmul(x[i], y[i]))

    z = np.stack(results)
    return z


def UnsecureMatmul(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape[1] == y.shape[0]

    M, K = x.shape
    _, N = y.shape

    z = np.zeros((M, N), dtype=np.int32)

    x_gpu = cuda.mem_alloc(x.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    z_gpu = cuda.mem_alloc(z.nbytes)

    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(y_gpu, y)

    NUM_THREADS = 128
    BN = 128
    BM = 128

    block_size = (NUM_THREADS, 1, 1)
    grid_size = (
        int(np.ceil(N / BN)), int(np.ceil(M / BM)))

    wt_matmul_s8x_s8y_s32z(x_gpu, y_gpu, z_gpu, np.int32(M), np.int32(N),
                           np.int32(K), block=block_size, grid=grid_size)
    cuda.Context.synchronize()

    cuda.memcpy_dtoh(z, z_gpu)

    x_gpu.free()
    y_gpu.free()
    z_gpu.free()

    return z


def BatchSecureMatmulFull(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 3 and y.ndim == 3, "Input tensors must be 3D (batched matrices)"
    assert x.shape[0] == y.shape[0], "Number of batches must be equal"

    batch_size, M, K = x.shape
    _, K, N = y.shape

    results = []

    for i in range(batch_size):
        results.append(SecureMatmulFull(x[i], y[i]))

    z = np.stack(results)
    return z


a_x = None
a_y = None
b_x = None
b_y = None
b_factor = None
key_inv = None


def SecureMatmulFull(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape[1] == y.shape[0]
    assert x.dtype == np.int8 and y.dtype == np.int8
    timer = st.SingletonTimer()

    B = 8

    M, K = x.shape
    _, N = y.shape

    # 8 bit -> 32 bit
    t = timer.start(tag='int8 to int32', category='int8 to int32')
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    timer.end(t)

    mod = 2**32

    t = timer.start(tag='gen. shift metadata', category='gen. shift metadata')
    # Generate metadata for shifting
    shift_row_sum_x = np.sum(x, axis=1, dtype=np.int32)
    shift_col_sum_y = np.sum(y, axis=0, dtype=np.int32)
    timer.end(t)

    amt = 2**(B-1)

    t = timer.start(tag='shift inputs', category='shift inputs')
    # Shift
    x += amt
    y += amt
    timer.end(t)

    # int32 -> uint32
    t = timer.start(tag='int32 to uint32', category='int32 to uint32')
    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    timer.end(t)

    # Generate metadata for decryption

    # Do precomputation only once
    t = timer.start(tag='precomputation', category='precomputation')
    global a_x, a_y, b_x, b_y, b_factor, key_inv
    if a_x is None:
        a_x = np.asanyarray(cip_cpp.GenerateKeySetA(mod, M), dtype=np.uint32)
        a_y = np.asanyarray(cip_cpp.GenerateKeySetA(mod, N), dtype=np.uint32)
        b_x = np.random.randint(0, mod - 1, (K), dtype=np.uint32)
        b_y = np.random.randint(0, mod - 1, (K), dtype=np.uint32)

        b_factor = np.dot(b_x, b_y)

        key_inv = np.asanyarray(
            cip_cpp.FindKeyInvModFull(a_x, a_y), dtype=np.uint32)
    timer.end(t)

    t = timer.start(tag='gen. decryption metadata', category='gen. decryption metadata')
    dec_row_sum_x = np.empty((M), dtype=np.uint32)
    dec_col_sum_y = np.empty((N), dtype=np.uint32)
    for i in range(M):
        dec_row_sum_x[i] = np.dot(b_y, x[i]) * a_x[i]
    for i in range(N):
        dec_col_sum_y[i] = np.dot(b_x, y[:, i]) * a_y[i]
    timer.end(t)

    t = timer.start(tag='encryption', category='encryption')
    cip_cpp.EncryptMatrix2DFull(x, a_x, b_x, False)
    cip_cpp.EncryptMatrix2DFull(y, a_y, b_y, True)
    timer.end(t)

    t = timer.start(tag='host to device', category='host to device')
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    timer.end(t)

    z_gpu = cp.matmul(x_gpu, y_gpu)
    t = timer.start(tag='gpu computation', category='gpu computation')
    z_gpu = cp.matmul(x_gpu, y_gpu)
    timer.end(t)
    t = timer.start(tag='device to host', category='device to host')
    z = cp.asnumpy(z_gpu)
    timer.end(t)

    t = timer.start(tag='decryption', category='decryption')
    cip_cpp.DecryptMatrix2DFull(
        z, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor)
    timer.end(t)

    t = timer.start(tag='uint32 to int32', category='uint32 to int32')
    z = z.astype(np.int32)
    timer.end(t)

    t = timer.start(tag='undo shift', category='undo shift')
    cip_cpp.UndoShift_int32(z, amt, K, shift_row_sum_x, shift_col_sum_y)
    timer.end(t)

    return z


def SecureMatmul(x: torch.Tensor, y: torch.Tensor):
    assert x.dtype == torch.int8 and y.dtype == torch.int8
    if x.dtype == torch.int8:
        B = 8
    timer = st.SingletonTimer()

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
