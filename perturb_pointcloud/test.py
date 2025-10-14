'''
test the modification to other people's code. 
Make sure they have same behaviors
'''

import numpy as np
from noise_corruption import fill_intensity
from corruption import fill_intensity as fill_intensity2
import timeit

def test_fill_intensity(num=100):
    for _ in range(num):
        N = np.random.randint(10000, 20000)
        N_all = N + np.random.randint(1000)
        pc = np.random.rand(N,4)
        pc_cor_xyz = np.random.rand(N_all,3)
        assert np.allclose(fill_intensity(pc, pc_cor_xyz), fill_intensity2(pc, pc_cor_xyz))

def performance_fill_intensity(num=100):
    N = np.random.randint(10000, 20000)
    N_all = N + np.random.randint(1000)
    pc = np.random.rand(N,4)
    pc_cor_xyz = np.random.rand(N_all,3)
    t1 = timeit.timeit(lambda: fill_intensity(pc, pc_cor_xyz), number=num)
    t2 = timeit.timeit(lambda: fill_intensity2(pc, pc_cor_xyz), number=num)
    print(f"t2:t1={t2/t1*100}%")

if __name__ == '__main__':
    test_fill_intensity()
    for i in range(10):
        performance_fill_intensity()