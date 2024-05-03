import numpy as np
import random
import singleton_timer as st
import csv

from cipher import cipher_light as cl
from cipher import cipher_unsecure as cu


def main():
    timer = st.SingletonTimer(False)

    batch_size = 10
    num_dim = 12
    num_iter = 100

    sbmm_light = cl.SBMM_Light()
    sbmm_unsecure = cu.SBMM_Unsecure()

    data = []

    for i in range(num_dim + 1):
        dim = 2**i

        warm_up_cnt = 3
        sbmm_unsecure.disable_timer()
        sbmm_light.disable_timer()

        for j in range(num_iter + warm_up_cnt):
            print(f"DIM = {dim}, iter = {j}")
            x = np.random.randint(-128, 128,
                                  (batch_size, 1, dim), dtype=np.int8)
            y = np.random.randint(-128, 128,
                                  (batch_size, dim, dim), dtype=np.int8)

            x2 = x.copy()
            y2 = y.copy()

            z2 = sbmm_unsecure(x2, y2)
            z = sbmm_light(x, y)

            assert np.array_equal(z, z2)

            warm_up_cnt -= 1
            if warm_up_cnt == 0:
                sbmm_light.enable_timer()

        raw_data = timer.display_summary()

        total_latencey = 0.0
        for _, v in raw_data.items():
            total_latencey += v

        partial_data = [f'B={batch_size},DIM={dim}']
        partial_data.append(raw_data['S8 to S32 (L)'])
        partial_data.append(raw_data['gen. shift metadata (L)'])
        partial_data.append(raw_data['shift inputs (L)'])
        partial_data.append(raw_data['S32 to U32 (L)'])
        partial_data.append(raw_data['gen. keys (L)'])
        partial_data.append(raw_data['gen. decryption metadata (L)'])
        partial_data.append(raw_data['encrypt (L)'])
        partial_data.append(raw_data['HtoD (L)'])
        partial_data.append(raw_data['gpu comp. (L)'])
        partial_data.append(raw_data['DtoH (L)'])
        partial_data.append(raw_data['decrypt (L)'])
        partial_data.append(raw_data['U32 to S32 (L)'])
        partial_data.append(raw_data['undo shift (L)'])
        partial_data.append(f'{total_latencey}')

        data.append(partial_data)

        timer.reset()

    f = open(f'batch_{batch_size}_decode_benchmark.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()


if __name__ == '__main__':
    main()
