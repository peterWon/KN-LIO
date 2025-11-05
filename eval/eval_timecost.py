import numpy as np
import os

logdir = '/media/wz/2C96A0A60155E8F8/Dataset/Portable/LOG/semi_knlio/vehicle_campus00/timecost/'
timecost_preprocess = np.loadtxt(os.path.join(logdir, 'preprocessing_cost_ms.txt')).astype(np.float32)
timecost_tracking = np.loadtxt(os.path.join(logdir, 'tracking_cost_ms.txt')).astype(np.float32)
timecost_mapping = np.loadtxt(os.path.join(logdir, 'mapping_cost_ms.txt')).astype(np.float32)
timecost_bundle_adjustment = np.loadtxt(os.path.join(logdir, 'bundle_adjustment_cost_ms.txt')).astype(np.float32)

print('Preprocessing:', np.mean(timecost_preprocess), 'ms')
print('Tracking:', np.mean(timecost_tracking), 'ms')
print('Mapping:', np.mean(timecost_mapping), 'ms')
# print('Bundle adjustment:', np.mean(timecost_bundle_adjustment), 'ms')

mean_timecost = np.mean(timecost_preprocess) + np.mean(timecost_tracking) + np.mean(timecost_mapping)
print('Total:', mean_timecost, 'ms')
print('FPS:', 1000.0/mean_timecost, 'fps')