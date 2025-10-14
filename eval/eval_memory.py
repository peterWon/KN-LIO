import numpy as np
import os

logdir = '/home/wz/codes/ros_ws/lios/pin-lio/experiments/viralxxx/fog-eee03/memory/'
CPU_memory = np.loadtxt(os.path.join(logdir, 'CPU_memory_mb.txt')).astype(np.float32)
GPU_memory = np.loadtxt(os.path.join(logdir, 'max_memory_allocated_mb.txt')).astype(np.float32)

print('CPU_memory:', np.max(CPU_memory) / 1024, 'Gb')
print('GPU_memory:', np.max(GPU_memory) / 1024, 'Gb')