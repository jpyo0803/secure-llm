import numpy as np
import singleton_timer as st
import cupy as cp

# Secure Batch Matrix Multiplication Light version


class SBMM_Unsecure:
    def __init__(self, timer_on: bool = False):
        self.timer_on = timer_on

    def __run(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 3 and y.ndim == 3
        assert x.shape[0] == y.shape[0]
        assert x.shape[2] == y.shape[1]
        assert x.dtype == np.int8 and y.dtype == np.int8

        if self.timer_on:
            timer = st.SingletonTimer()

        batch_size, M, K = x.shape
        _, _, N = y.shape

        if self.timer_on:
            t = timer.start(tag='S8 to S32 (U)',
                            category='S8 to S32 (U)')
        x = x.astype(np.int32, copy=False)
        y = y.astype(np.int32, copy=False)
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='HtoD (U)',
                            category='HtoD (U)')

        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)

        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='gpu comp. (U)', category='gpu comp. (U)')
        z_gpu = cp.matmul(x_gpu, y_gpu)
        if self.timer_on:
            timer.end(t)
        if self.timer_on:
            t = timer.start(tag='DtoH (U)', category='DtoH (U)')
        z = cp.asnumpy(z_gpu)
        if self.timer_on:
            timer.end(t)

        return z

    def disable_timer(self):
        self.timer_on = False
    
    def enable_timer(self):
        self.timer_on = True
        
    def __call__(self, x: np.ndarray, y: np.ndarray):
        return self.__run(x, y)
