import singleton_timer as st
import numpy as np
import cupy as cp
import nvtx
import concurrent.futures
from enum import Enum
import ntl
import torch

from multiprocessing import Pool


class CipherType(Enum):
    Unsecure = 1
    Mine = 2
    Slalom = 3


def decrypt_matrix_2d_full(in_matrix, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor):
    adjusted_matrix = in_matrix - \
        (dec_row_sum_x[:, :, np.newaxis] + dec_col_sum_y[:, np.newaxis, :])
    adjusted_matrix -= b_factor[:, np.newaxis, np.newaxis]
    adjusted_matrix *= key_inv
    return adjusted_matrix


def decrypt_matrix_2d_full_torch(in_matrix, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor):
    adjusted_matrix = in_matrix - \
        (dec_row_sum_x[:, :, None] + dec_col_sum_y[:, None, :])
    adjusted_matrix -= b_factor[:, None, None]
    adjusted_matrix *= key_inv
    return adjusted_matrix


def compute_decryption_metadata(x, y, b_x, b_y, a_x, a_y):
    batch_size, M, _ = x.shape
    _, _, N = y.shape
    dec_row_sum_x = np.empty((batch_size, M), dtype=np.uint32)
    dec_col_sum_y = np.empty((batch_size, N), dtype=np.uint32)
    for i in range(batch_size):
        dec_row_sum_x[i] = np.einsum(
            'i,ij,j->i', a_x[i], x[i], b_y[i], optimize=True)
        dec_col_sum_y[i] = np.einsum(
            'i,ij,j->i', a_y[i], y[i].T, b_x[i], optimize=True)

    return dec_row_sum_x, dec_col_sum_y


def compute_decryption_metadata_torch(x, y, b_x, b_y, a_x, a_y):
    # Uncomment these print statements to debug and check the shapes if necessary.
    # print(b_y.shape)
    # print(x.shape)
    # print(a_x.shape)
    # print("----------")
    batch_size, M, _ = x.shape
    _, _, N = y.shape

    # Use torch.empty to allocate uninitialized tensors, similar to np.empty.
    dec_row_sum_x = torch.empty((batch_size, M), dtype=torch.int64)
    dec_col_sum_y = torch.empty((batch_size, N), dtype=torch.int64)

    for i in range(batch_size):
        # Using torch.einsum to perform the tensor contractions.
        # Note: PyTorch does not have an `optimize` parameter in `einsum`, but PyTorch's einsum is generally well optimized.
        dec_row_sum_x[i] = torch.einsum(
            'i,ij,j->i', a_x[i], x[i], b_y[i])
        dec_col_sum_y[i] = torch.einsum(
            'i,ij,j->i', a_y[i], y[i].T, b_x[i])

    return dec_row_sum_x, dec_col_sum_y


def encrypt_matrix_full(args):
    matrix, a, b, vertical = args
    M, N = matrix.shape
    print(a.shape)
    if vertical:
        a = a.reshape(1, N)
        b = b.reshape(M, 1)
    else:
        a = a.reshape(M, 1)
        b = b.reshape(1, N)

    matrix *= a
    matrix += b


def encrypt_tensor_full(tensor, a, b, vertical):
    batch_size, M, N = tensor.shape
    args = [(tensor[i], a[i], b[i], vertical) for i in range(batch_size)]

    with Pool(batch_size) as pool:
        pool.map(encrypt_matrix_full, args)


def encrypt_matrix_2d_full(in_matrix, a, b, vertical):
    batch_size, M, N = in_matrix.shape

    result = in_matrix
    if vertical:
        a = a.reshape(batch_size, 1, N)
        b = b.reshape(batch_size, M, 1)
    else:
        a = a.reshape(batch_size, M, 1)
        b = b.reshape(batch_size, 1, N)

    result *= a
    result += b

    return result


def encrypt_matrix_2d_full_torch(in_matrix, a, b, vertical):
    batch_size, M, N = in_matrix.shape
    result = in_matrix

    if vertical:
        a = a.view(batch_size, 1, N)
        b = b.view(batch_size, M, 1)
    else:
        a = a.view(batch_size, M, 1)
        b = b.view(batch_size, 1, N)

    result *= a
    result += b

    return result

# def undo_shift(in_matrix, amt, K, row_sum_x, col_sum_y):
#     in_matrix -= amt * (row_sum_x[:, :, np.newaxis] +
#                         col_sum_y[:, np.newaxis, :] + K * amt)
#     return in_matrix


def undo_shift_optimized(in_matrix, amt, K, row_sum_x, col_sum_y):
    adjustment = row_sum_x[:, :, np.newaxis] * amt + K * amt**2
    in_matrix -= adjustment
    in_matrix -= col_sum_y[:, np.newaxis, :] * amt
    return in_matrix


def BatchSecureMatmul(x: np.ndarray, y: np.ndarray, scale: np.ndarray = None, timer_on=False):
    return BatchSecureMatmulMine(x, y, timer_on)


def BSM_S8S8F32(x: np.ndarray, y: np.ndarray, scale: np.float32 = None, cipher_type: CipherType = CipherType.Mine, timer_on: bool = False):
    if cipher_type is CipherType.Mine:
        result = BatchSecureMatmulMine(x, y, timer_on)
        return result * scale


blind_factor = None


def BatchSecureMatmulSlalom(x: np.ndarray, y: np.ndarray, timer_on: bool = False):
    pass


def BatchSecureMatmulLight(x: np.ndarray, y: np.ndarray, timer_on=False):
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape[0] == y.shape[0], "Number of batches must be equal"
    assert x.shape[2] == y.shape[1]
    assert x.dtype == np.int8 and y.dtype == np.int8
    if timer_on:
        timer = st.SingletonTimer()

    B = 8

    batch_size, M, K = x.shape
    _, _, N = y.shape

    if timer_on:
        t = timer.start(tag='int8 to int32 (light)',
                        category='int8 to int32 (light)')
    x = x.astype(np.int32, copy=False)
    y = y.astype(np.int32, copy=False)
    if timer_on:
        timer.end(t)

    mod = 2**32

    amt = 2**(B-1)
    if timer_on:
        t = timer.start(tag='gen. shift metadata (light)',
                        category='gen. shift metadata (light)')
    # Generate metadata for shifting
    shift_row_sum_x = np.sum(x, axis=2, dtype=np.int32)
    shift_col_sum_y = np.sum(y, axis=1, dtype=np.int32)

    undo_shift_factor = shift_row_sum_x[:, :, np.newaxis] * amt + \
        K * amt**2 + shift_col_sum_y[:, np.newaxis, :] * amt
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='shift inputs (light)',
                        category='shift inputs (light)')
    # Shift
    x += amt
    y += amt
    if timer_on:
        timer.end(t)

    # int32 -> uint32
    if timer_on:
        t = timer.start(tag='int32 to uint32 (light)',
                        category='int32 to uint32 (light)')
    x = x.view(np.uint32)
    y = y.view(np.uint32)
    if timer_on:
        timer.end(t)

    # keys are shared among batches
    if timer_on:
        t = timer.start(tag='generate keys (light)',
                        category='generate keys (light)')
    b_x = np.random.randint(
        0, mod - 1, size=(K), dtype=np.uint32)
    b_y = np.random.randint(
        0, mod - 1, size=(K), dtype=np.uint32)
    a_x = np.array(ntl.ntl_cpp.GenerateRandomCoprimeLessthanN_uint64(mod), dtype=np.uint32)
    a_y = np.array(ntl.ntl_cpp.GenerateRandomCoprimeLessthanN_uint64(mod), dtype=np.uint32)
    ax_ay_inv = np.array(ntl.FindKeyInvModNonPrime(a_x * a_y, mod), dtype=np.uint32)
    b_factor = np.dot(b_x, b_y)

    # assert (a_x *a_y*ax_ay_inv % mod) == 1
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='gen. decryption metadata (light)',
                        category='gen. decryption metadata (light)')
    dec_row_sum_x, dec_col_sum_y = compute_decryption_metadata_light(
        x, y, a_x, a_y, b_x, b_y)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='encryption (light)',
                        category='encryption (light)')

    x = encrypt_tensor_light(x, a_x, b_x, False)
    y = encrypt_tensor_light(y, a_y, b_y, True)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='host to device (light)', category='host to device (light)')
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='gpu computation (light)', category='gpu computation (light)')
    z_gpu = cp.matmul(x_gpu, y_gpu)
    if timer_on:
        timer.end(t)
    if timer_on:
        t = timer.start(tag='device to host (light)', category='device to host (light)')
    z = cp.asnumpy(z_gpu)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='decryption (light)', category='decryption (light)')

    # z = decrypt_matrix_2d_full(
    #     z, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor)
    print("beofe dec z : ", z)
    print("ax_ay_inv: ", ax_ay_inv)
    print("dec row x: ", dec_row_sum_x)
    print("dec col y: ", dec_col_sum_y)
    print("b_factor: ", b_factor)
    z = decrypt_tensor_light(z, ax_ay_inv, dec_row_sum_x, dec_col_sum_y, b_factor)
    print("after dec z : ", z)

    if timer_on:
        timer.end(t)
    if timer_on:
        t = timer.start(tag='uint32 to int32 (light)',
                        category='uint32 to int32 (light)')
    z = z.view(np.int32)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='undo shift (light)',
                        category='undo shift (light)')
    z -= undo_shift_factor
    if timer_on:
        timer.end(t)
    return z

a_x = None
a_y = None
b_x = None
b_y = None
b_factor = None
key_inv = None


def BatchSecureMatmulMine(x: np.ndarray, y: np.ndarray, timer_on=False):
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape[0] == y.shape[0], "Number of batches must be equal"
    assert x.shape[2] == y.shape[1]
    assert x.dtype == np.int8 and y.dtype == np.int8
    if timer_on:
        timer = st.SingletonTimer()

    B = 8

    batch_size, M, K = x.shape
    _, _, N = y.shape

    # 8 bit -> 32 bit, here must be copied
    if timer_on:
        t = timer.start(tag='int8 to int32', category='int8 to int32')
    x = x.astype(np.int32, copy=False)
    y = y.astype(np.int32, copy=False)
    if timer_on:
        timer.end(t)

    mod = 2**32

    amt = 2**(B-1)
    if timer_on:
        t = timer.start(tag='gen. shift metadata',
                        category='gen. shift metadata')
    # Generate metadata for shifting
    shift_row_sum_x = np.sum(x, axis=2, dtype=np.int32)
    shift_col_sum_y = np.sum(y, axis=1, dtype=np.int32)

    print("x: ", x)
    print("y: ", x)
    print("shift row sum x: ", shift_row_sum_x)
    print("shift col sum y: ", shift_col_sum_y)
    undo_shift_factor = shift_row_sum_x[:, :, np.newaxis] * amt + \
        K * amt**2 + shift_col_sum_y[:, np.newaxis, :] * amt
    print("undo shift factor : ", undo_shift_factor)
    if timer_on:
        timer.end(t)
    assert False

    if timer_on:
        t = timer.start(tag='shift inputs', category='shift inputs')

    # Shift
    x += amt
    y += amt
    if timer_on:
        timer.end(t)

    # int32 -> uint32
    if timer_on:
        t = timer.start(tag='int32 to uint32', category='int32 to uint32')
    x = x.view(np.uint32)
    y = y.view(np.uint32)
    if timer_on:
        timer.end(t)

    # Generate metadata for decryption

    # Do precomputation only once
    # t = timer.start(tag='precomputation', category='precomputation')
    global a_x, a_y, b_x, b_y, b_factor, key_inv
    if a_x is None:
        a_x = np.empty((batch_size, M), dtype=np.uint32)
        a_y = np.empty((batch_size, N), dtype=np.uint32)
        b_x = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_y = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_factor = np.einsum('ij,ij->i', b_x, b_y)
        key_inv = np.empty((batch_size, M, N), dtype=np.uint32)
        for i in range(batch_size):
            a_x[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, M), dtype=np.uint32)
            a_y[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, N), dtype=np.uint32)
            key_inv[i] = np.asanyarray(
                cip_cpp.FindKeyInvModFull(a_x[i], a_y[i]), dtype=np.uint32)
    # timer.end(t)

    if timer_on:
        t = timer.start(tag='gen. decryption metadata',
                        category='gen. decryption metadata')

    dec_row_sum_x, dec_col_sum_y = compute_decryption_metadata(
        x, y, b_x, b_y, a_x, a_y)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='encryption', category='encryption')

    x = encrypt_matrix_2d_full(x, a_x, b_x, False)
    y = encrypt_matrix_2d_full(y, a_y, b_y, True)

    if timer_on:
        timer.end(t)
    if timer_on:
        # nvtx.push_range("host to device")
        t = timer.start(tag='host to device', category='host to device')
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()

    if timer_on:
        # nvtx.push_range("gpu computation")
        t = timer.start(tag='gpu computation', category='gpu computation')
    z_gpu = cp.matmul(x_gpu, y_gpu)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()
    if timer_on:
        # nvtx.push_range("device to host")
        t = timer.start(tag='device to host', category='device to host')
    z = cp.asnumpy(z_gpu)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()

    if timer_on:
        t = timer.start(tag='decryption', category='decryption')

    z = decrypt_matrix_2d_full(
        z, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='uint32 to int32', category='uint32 to int32')
    z = z.view(np.int32)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='undo shift', category='undo shift')
    z -= undo_shift_factor
    if timer_on:
        timer.end(t)

    return z


a_x2 = None
a_y2 = None
b_x2 = None
b_y2 = None
b_factor2 = None
key_inv2 = None


def BatchSecureMatmulMineTorch(x: torch.Tensor, y: torch.Tensor, timer_on=False):
  #  assert x.ndim == 3 and y.ndim == 3
  #  assert x.shape[0] == y.shape[0], "Number of batches must be equal"
  #  assert x.shape[2] == y.shape[1]
    assert x.dtype == torch.int8 and y.dtype == torch.int8
    if timer_on:
        timer = st.SingletonTimer()

    B = 8

    batch_size, M, K = x.shape
    _, _, N = y.shape

    # 8 bit -> 32 bit, here must be copied
    if timer_on:
        t = timer.start(tag='int8 to int32 (torch)',
                        category='int8 to int32 (torch)')
    x = x.to(torch.int32)
    y = y.to(torch.int32)
    if timer_on:
        timer.end(t)

    mod = 2**32

    amt = 2**(B-1)
    if timer_on:
        t = timer.start(tag='gen. shift metadata (torch)',
                        category='gen. shift metadata (torch)')
    # Generate metadata for shifting
    shift_row_sum_x = torch.sum(x, dim=2)
    shift_col_sum_y = torch.sum(y, dim=1)

    undo_shift_factor = shift_row_sum_x[:, :, None] * amt + \
        K * amt**2 + shift_col_sum_y[:, None, :] * amt
    if timer_on:
        timer.end(t)

    # undo_shift_factor = undo_shift_factor.numpy()

    if timer_on:
        t = timer.start(tag='shift inputs (torch)',
                        category='shift inputs (torch)')

    # Shift
    x += amt
    y += amt

    if timer_on:
        timer.end(t)

    # int32 -> uint32
    if timer_on:
        t = timer.start(tag='int32 to uint32 (torch)',
                        category='int32 to uint32 (torch)')

   # x = x.to(torch.uint32)
   # y = y.to(torch.uint32)

    if timer_on:
        timer.end(t)

    # Generate metadata for decryption

    # Do precomputation only once
    # t = timer.start(tag='precomputation', category='precomputation')
    global a_x2, a_y2, b_x2, b_y2, b_factor2, key_inv2
    if a_x2 is None:
        a_x2 = np.empty((batch_size, M), dtype=np.uint32)
        a_y2 = np.empty((batch_size, N), dtype=np.uint32)
        b_x2 = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_y2 = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_factor2 = np.einsum('ij,ij->i', b_x2, b_y2)
        key_inv2 = np.empty((batch_size, M, N), dtype=np.uint32)
        for i in range(batch_size):
            a_x2[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, M), dtype=np.uint32)
            a_y2[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, N), dtype=np.uint32)
            key_inv2[i] = np.asanyarray(
                cip_cpp.FindKeyInvModFull(a_x2[i], a_y2[i]), dtype=np.uint32)
        a_x2 = torch.from_numpy(a_x2.astype(np.int64))
        a_y2 = torch.from_numpy(a_y2.astype(np.int64))
        b_x2 = torch.from_numpy(b_x2.astype(np.int64))
        b_y2 = torch.from_numpy(b_y2.astype(np.int64))
        key_inv2 = torch.from_numpy(key_inv2.astype(np.int64))
    # timer.end(t)

    if timer_on:
        t = timer.start(tag='gen. decryption metadata (torch)',
                        category='gen. decryption metadata (torch)')

    dec_row_sum_x, dec_col_sum_y = compute_decryption_metadata_torch(
        x, y, b_x2, b_y2, a_x2, a_y2)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='encryption (torch)',
                        category='encryption (torch)')

    x = encrypt_matrix_2d_full_torch(x, a_x2, b_x2, False)
    y = encrypt_matrix_2d_full_torch(y, a_y2, b_y2, True)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='convert tensor to numpy (torch)',
                        category='convert tensor to numpy (torch)')
    x = x.numpy()
    y = y.numpy()
    if timer_on:
        timer.end(t)

    if timer_on:
        # nvtx.push_range("host to device")
        t = timer.start(tag='host to device (torch)',
                        category='host to device (torch)')
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()

    if timer_on:
        # nvtx.push_range("gpu computation")
        t = timer.start(tag='gpu computation (torch)',
                        category='gpu computation (torch)')
    z_gpu = cp.matmul(x_gpu, y_gpu)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()
    if timer_on:
        # nvtx.push_range("device to host")
        t = timer.start(tag='device to host (torch)',
                        category='device to host (torch)')
    z = cp.asnumpy(z_gpu)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()

    if timer_on:
        t = timer.start(tag='convert numpy to tensor (torch)',
                        category='convert numpy to tensor (torch)')
    z = torch.from_numpy(z)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='decryption (torch)',
                        category='decryption (torch)')

    z = decrypt_matrix_2d_full_torch(
        z, key_inv2, dec_row_sum_x, dec_col_sum_y, b_factor2)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='uint32 to int32 (torch)',
                        category='uint32 to int32 (torch)')
    z = z.to(torch.int32)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='undo shift (torch)',
                        category='undo shift (torch)')
    z -= undo_shift_factor
    if timer_on:
        timer.end(t)

    return z


a_x3 = None
a_y3 = None
b_x3 = None
b_y3 = None
b_factor3 = None
key_inv3 = None


def BatchSecureMatmulMineMixed(x: torch.Tensor, y: torch.Tensor, timer_on=False):
  #  assert x.ndim == 3 and y.ndim == 3
  #  assert x.shape[0] == y.shape[0], "Number of batches must be equal"
  #  assert x.shape[2] == y.shape[1]
    assert x.dtype == torch.int8 and y.dtype == torch.int8
    if timer_on:
        timer = st.SingletonTimer()

    B = 8

    batch_size, M, K = x.shape
    _, _, N = y.shape

    # 8 bit -> 32 bit, here must be copied
    if timer_on:
        t = timer.start(tag='int8 to int32 (mixed)',
                        category='int8 to int32 (mixed)')
    x = x.to(torch.int32)
    y = y.to(torch.int32)
    if timer_on:
        timer.end(t)

    mod = 2**32

    amt = 2**(B-1)
    if timer_on:
        t = timer.start(tag='gen. shift metadata (mixed)',
                        category='gen. shift metadata (mixed)')
    # Generate metadata for shifting
    shift_row_sum_x = torch.sum(x, dim=2)
    shift_col_sum_y = torch.sum(y, dim=1)

    undo_shift_factor = shift_row_sum_x[:, :, None] * amt + \
        K * amt**2 + shift_col_sum_y[:, None, :] * amt
    if timer_on:
        timer.end(t)

    # undo_shift_factor = undo_shift_factor.numpy()

    if timer_on:
        t = timer.start(tag='shift inputs (mixed)',
                        category='shift inputs (mixed)')

    # Shift
    x += amt
    y += amt

    if timer_on:
        timer.end(t)

    # int32 -> uint32
    if timer_on:
        t = timer.start(tag='int32 to uint32 (mixed)',
                        category='int32 to uint32 (mixed)')

   # x = x.to(torch.uint32)
   # y = y.to(torch.uint32)

    if timer_on:
        timer.end(t)

    # Generate metadata for decryption

    # Do precomputation only once
    # t = timer.start(tag='precomputation', category='precomputation')
    global a_x3, a_y3, b_x3, b_y3, b_factor3, key_inv3
    if a_x3 is None:
        a_x3 = np.empty((batch_size, M), dtype=np.uint32)
        a_y3 = np.empty((batch_size, N), dtype=np.uint32)
        b_x3 = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_y3 = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_factor3 = np.einsum('ij,ij->i', b_x3, b_y3)
        key_inv3 = np.empty((batch_size, M, N), dtype=np.uint32)
        for i in range(batch_size):
            a_x3[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, M), dtype=np.uint32)
            a_y3[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, N), dtype=np.uint32)
            key_inv3[i] = np.asanyarray(
                cip_cpp.FindKeyInvModFull(a_x3[i], a_y3[i]), dtype=np.uint32)
    # timer.end(t)

    if timer_on:
        t = timer.start(tag='gen. decryption metadata (mixed)',
                        category='gen. decryption metadata (mixed)')

    dec_row_sum_x, dec_col_sum_y = compute_decryption_metadata(
        x, y, b_x3, b_y3, a_x3, a_y3)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='encryption (mixed)',
                        category='encryption (mixed)')

    x = encrypt_matrix_2d_full(x, a_x3, b_x3, False)
    y = encrypt_matrix_2d_full(y, a_y3, b_y3, True)

    if timer_on:
        timer.end(t)
    if timer_on:
        # nvtx.push_range("host to device")
        t = timer.start(tag='host to device (mixed)',
                        category='host to device (mixed)')
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()

    if timer_on:
        # nvtx.push_range("gpu computation")
        t = timer.start(tag='gpu computation (mixed)',
                        category='gpu computation (mixed)')
    z_gpu = cp.matmul(x_gpu, y_gpu)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()
    if timer_on:
        # nvtx.push_range("device to host")
        t = timer.start(tag='device to host (mixed)',
                        category='device to host (mixed)')
    z = cp.asnumpy(z_gpu)
    if timer_on:
        timer.end(t)
        # nvtx.pop_range()

    if timer_on:
        t = timer.start(tag='decryption (mixed)',
                        category='decryption (mixed)')

    z = decrypt_matrix_2d_full(
        z, key_inv3, dec_row_sum_x, dec_col_sum_y, b_factor3)

    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='uint32 to int32 (mixed)',
                        category='uint32 to int32 (mixed)')
    z = z.view(np.int32)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='undo shift (mixed)',
                        category='undo shift (mixed)')
    z -= undo_shift_factor
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='convert numpy to tensor (mixed)',
                        category='convert numpy to tensor (mixed)')
    z = torch.from_numpy(z)
    if timer_on:
        timer.end(t)

    return torch.z


def BatchUnsecureMatmul(x: np.ndarray, y: np.ndarray, timer_on=False):
    assert x.ndim == 3
    assert x.shape[0] == y.shape[0], "Number of batches must be equal"
    assert x.shape[2] == y.shape[1]
    assert x.dtype == np.int8 and y.dtype == np.int8

    if y.ndim == 2:
        y_gpu = y_gpu[np.newaxis, :, :]

    if timer_on:
        timer = st.SingletonTimer()

    if timer_on:
        t = timer.start(tag='int8 to int32 (unsecure)',
                        category='int8 to int32 (unsecure)')
    x = x.astype(np.int32, copy=False)
    y = y.astype(np.int32, copy=False)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='host to device (unsecure)',
                        category='host to device (unsecure)')
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='gpu computation (unsecure)',
                        category='gpu computation (unsecure)')
    z_gpu = cp.matmul(x_gpu, y_gpu)
    if timer_on:
        timer.end(t)

    if timer_on:
        t = timer.start(tag='device to host (unsecure)',
                        category='device to host (unsecure)')
    z = cp.asnumpy(z_gpu)
    if timer_on:
        timer.end(t)

    return z


def BatchUnsecureLinear(x: np.ndarray, y: np.ndarray, bias: np.ndarray, timer_on=False):
    pass


def compute_decryption_metadata_light(x, y, a_x, a_y, b_x, b_y):
    batch_size, M, _ = x.shape
    _, _, N = y.shape

    # a_x, a_y: (1,)
    # b_x, b_y: (K,)
    # x: (batch_size, M, K)
    # y: (batch_size, K, N)

    dec_row_sum_x = np.sum(x * b_y, axis=2, dtype=np.uint32) * a_x
    dec_col_sum_y = np.sum(y * b_x[np.newaxis, :, np.newaxis], axis=1, dtype=np.uint32) * a_y
    return dec_row_sum_x, dec_col_sum_y

def encrypt_tensor_light(tensor, a, b, vertical):
    # tensor: (batch_size, M, K), if horizontal
    # tensor: (batch_size, K, N), if vertical
    # a: (1,)
    # b: (K,)

    if vertical:
        return np.moveaxis(tensor, -1, -2) * a + b
    return tensor * a + b

def decrypt_tensor_light(tensor, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor):
    # tensor: (batch_size, M, N)
    # key_inv: (1, )
    # dec_row_sum_x: (batch_size, M)
    # dec_col_sum_y: (batch_size, N)
    # b_factor: (1,)
    out = tensor - dec_row_sum_x[:, :, np.newaxis] - dec_col_sum_y[:, np.newaxis, :]
    out -= b_factor
    out *= key_inv
    return out