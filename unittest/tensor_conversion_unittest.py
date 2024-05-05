import cipher
import numpy as np
import cipher.tensor_conversion
import torch
import cupy as cp
import random
import singleton_timer as st

num_test = 13
M = 2**13
N = 2**13

def test_from_numpy_to_cupy():
    timer = st.SingletonTimer()

    pass_cnt = 0
    for i in range(num_test):
        x = np.random.randint(-128, 128, (M, N), dtype=np.int32)

        t = timer.start(tag='from_numpy_to_cupy',
                        category='from_numpy_to_cupy', exclude=i < 3)
        y = cipher.tensor_conversion.FromNumpyToCupy(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_numpy_to_cupy' passed {pass_cnt} / {num_test}")


def test_from_cupy_to_numpy():
    timer = st.SingletonTimer()

    pass_cnt = 0
    for i in range(num_test):
        x = cp.random.randint(-128, 128, (M, N), dtype=cp.int32)

        t = timer.start(tag='from_cupy_to_numpy',
                        category='from_cupy_to_numpy', exclude=i < 3)
        y = cipher.tensor_conversion.FromCupyToNumpy(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_cupy_to_numpy' passed {pass_cnt} / {num_test}")


def test_from_torch_to_cupy():
    timer = st.SingletonTimer()

    pass_cnt = 0
    for i in range(num_test):
        x = torch.randint(-128, 128, (M, N),
                          dtype=torch.int32).to(torch.device('cuda:0'))

        t = timer.start(tag='from_torch_to_cupy',
                        category='from_torch_to_cupy', exclude=i < 3)
        y = cipher.tensor_conversion.FromTorchToCupy(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_torch_to_cupy' passed {pass_cnt} / {num_test}")


def test_from_cupy_to_torch():
    timer = st.SingletonTimer()

    pass_cnt = 0
    for i in range(num_test):
        x = cp.random.randint(-128, 128, (M, N), dtype=cp.int32)

        t = timer.start(tag='from_cupy_to_torch',
                        category='from_cupy_to_torch', exclude=i < 3)
        y = cipher.tensor_conversion.FromCupyToTorch(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_cupy_to_torch' passed {pass_cnt} / {num_test}")


def test_from_torch_to_numpy():
    timer = st.SingletonTimer()

    pass_cnt = 0

    for i in range(num_test):
        x = torch.randint(-128, 128, (M, N),
                          dtype=torch.int32, device=torch.device('cpu'))

        t = timer.start(tag='from_torch_to_numpy',
                        category='from_torch_to_numpy', exclude=i < 3)
        y = cipher.tensor_conversion.FromTorchToNumpy(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_torch_to_numpy' passed {pass_cnt} / {num_test}")


def test_from_numpy_to_torch():
    timer = st.SingletonTimer()

    pass_cnt = 0

    for i in range(num_test):
        x = np.random.randint(-128, 128, (M, N), dtype=np.int32)

        t = timer.start(tag='from_numpy_to_torch',
                        category='from_numpy_to_torch', exclude=i < 3)
        y = cipher.tensor_conversion.FromNumpyToTorch(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_numpy_to_torch' passed {pass_cnt} / {num_test}")

def test_from_cpu_to_gpu_torch():
    timer = st.SingletonTimer()

    pass_cnt = 0

    for i in range(num_test):
        x = torch.randint(-128, 128, (M, N),
                          dtype=torch.int32, device=torch.device('cpu'))

        t = timer.start(tag='from_cpu_to_gpu_torch',
                        category='from_cpu_to_gpu_torch', exclude=i < 3)
        y = cipher.tensor_conversion.FromCpuToGPUTorch(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_cpu_to_gpu_torch' passed {pass_cnt} / {num_test}")

def test_from_gpu_to_cpu_torch():
    timer = st.SingletonTimer()

    pass_cnt = 0

    for i in range(num_test):
        x = torch.randint(-128, 128, (M, N),
                          dtype=torch.int32, device=torch.device('cuda:0'))

        t = timer.start(tag='from_gpu_to_cpu_torch',
                        category='from_gpu_to_cpu_torch', exclude=i < 3)
        y = cipher.tensor_conversion.FromGpuToCpuTorch(x)
        timer.end(t)

        pass_cnt += 1
    print(f"'from_gpu_to_cpu_torch' passed {pass_cnt} / {num_test}")


if __name__ == '__main__':
    test_from_torch_to_cupy()
    test_from_cupy_to_torch()
    test_from_cupy_to_numpy()
    test_from_numpy_to_cupy()
    test_from_torch_to_numpy()
    test_from_numpy_to_torch()
    test_from_cpu_to_gpu_torch()
    test_from_gpu_to_cpu_torch()

    timer = st.SingletonTimer()
    timer.display_summary()
