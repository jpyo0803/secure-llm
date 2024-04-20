import ntl
import build.cipher_cpp as cip_cpp
import torch
import singleton_timer as st
import numpy as np
import random


def SecureMatmul(x: torch.Tensor, y: torch.Tensor, B: int):
    assert B <= 8

    timer = st.SingletonTimer()

    assert x.size()[1] == y.size()[0]
    M, K = x.size()
    _, N = y.size()

    mod = 2 ** 32
    a1, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    b1 = random.randint(0, mod)
    a2, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    b2 = random.randint(0, mod)
    # print(f'a1 = {a1}, b1 = {b1}, a2 = {a2}, b2 = {b2}')

    # Type must be int32 to properly handle negatives
    x_np = x.numpy().astype(np.int32)
    y_np = y.numpy().astype(np.int32)

    # Compute Row sum and Col sum for shifting, Need copy to be hidden
    shift_row_sum_x = cip_cpp.SumByRow_int32(x_np)
    shift_col_sum_y = cip_cpp.SumByCol_int32(y_np)

    # Now need to shift
    amt = 2**(B-1)

    cip_cpp.Shift_int32(x_np, amt)
    cip_cpp.Shift_int32(y_np, amt)

    # Compute Row sum and Col sum for Decryption (can be hidden)
    dec_row_sum_x = cip_cpp.SumByRow_int32(x_np)
    dec_col_sum_y = cip_cpp.SumByCol_int32(y_np)

    # Need to encrypt
    x_np = x_np.astype(np.uint32)
    y_np = y_np.astype(np.uint32)

    cip_cpp.EncryptMatrix2D(x_np, a1, b1, mod)
    cip_cpp.EncryptMatrix2D(y_np, a2, b2, mod)

    z_np = np.matmul(x_np, y_np)

    # Compute Key inv mod
    a_12_key_inv = ntl.FindKeyInvMod(a1 * a2, mod)

    cip_cpp.DecryptMatrix2D(z_np, K, a1, b1, a2, b2,
                            a_12_key_inv, mod, dec_row_sum_x, dec_col_sum_y)

    z_np = z_np.astype(np.int32)
    cip_cpp.UndoShift_int32(z_np, amt, K, shift_row_sum_x, shift_col_sum_y)
    z = torch.from_numpy(z_np)
    return z
