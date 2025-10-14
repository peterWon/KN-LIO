import sys
sys.path.insert(0, '/home/wz/codes/ros_ws/lios/pin-lio/perturb_pointcloud')
from pylisa.lisa import Lisa 
# from LISA.python_old.atmos_models import *
from fog_simulation import simulate_fog, ParameterSet
from wet_ground.augmentation import ground_water_augmentation
import density_corruption
import noise_corruption
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Corruption:
    # Lisa initialization is slow. we only create one instance for all cases
    rain_model = Lisa(atm_model = 'rain') #C++
    snow_model = Lisa(atm_model = 'snow') #C++
    # rain_model = LISA(atm_model = 'rain')
    # snow_model = LISA(atm_model = 'snow')

    @staticmethod
    def rain(scan, severity):
        severity = [5.0, 15.0, 50.0, 150.0, 500.0][severity - 1]
        return Corruption.rain_model.augment(scan, severity)[:, :4]

    @staticmethod
    def snow(scan, severity):
        severity = [0.5, 1.5, 5.0, 15.0, 50.0][severity - 1]
        return Corruption.snow_model.augment(scan, severity)[:, :4]

    @staticmethod
    def fog(scan, severity):
        # expected lidar intensity is 0-255
        severity = [0.005, 0.01, 0.02, 0.05, 0.1][severity - 1]
        scan[:, -1] *= 255
        p = ParameterSet(alpha=severity, gamma=0.000001)
        aug_scan, _, _ = simulate_fog(p, scan, 10)
        aug_scan[:, -1] /= 255
        return aug_scan
    
    @staticmethod
    def wet_ground_model(scan, water_height):
        scan[:, -1] *= 255
        labels = np.zeros([scan.shape[0], 1])
        aug_scan = np.hstack([scan, labels])
        aug_scan = ground_water_augmentation(aug_scan, water_height=water_height * 1e-3, debug=False)[:, :4]
        aug_scan[:, -1] /= 255
        return aug_scan
    
    @staticmethod
    def rain_wet_ground(scan, severity):
        water_height = [5.0, 10.0, 15.0, 25.0, 50.0][severity - 1]
        return Corruption.wet_ground_model(Corruption.rain(scan, severity), water_height)
    
    @staticmethod
    def snow_wet_ground(scan, severity):
        water_height = [0.5, 1, 1.5, 2.5, 5.0][severity - 1]
        return Corruption.wet_ground_model(Corruption.snow(scan, severity), water_height)

def fill_intensity(pc, pc_cor_xyz, n_b=5):
    N, _ = pc.shape
    N_all, _ = pc_cor_xyz.shape
    if N == N_all:
        return np.hstack((pc_cor_xyz, pc[:, 3].reshape([-1,1])))
    else:
        pc_cor = np.hstack((pc_cor_xyz, np.zeros((N_all,1))))
        pc_cor[:N,3] = pc[:,3]

        nbrs = NearestNeighbors(n_neighbors=n_b, algorithm='auto').fit(pc[:,:3])
        distances, indices = nbrs.kneighbors(pc_cor[N:,:3])
        nearest_intensities = pc[indices.flatten(), 3].reshape(indices.shape)
        pc_cor[N:, 3] = np.mean(nearest_intensities, axis=1)
        return pc_cor

def noise_template(func):
    def func_wrapper(scan, severity):
        crp_xyz = func(scan[:, 0:3], severity)
        crp = fill_intensity(scan, crp_xyz)
        return crp
    return func_wrapper

def density_template(func):
    return func

def add_corruptions(cls, module, func_names, func_template):
    for name in func_names:
        func = func_template(getattr(module, name))
        setattr(cls, name, staticmethod(func))

add_corruptions(Corruption, density_corruption, 
                ['cutout', 'density_inc', 'density_dec', 'beam_del', 'layer_del'],
                density_template)

add_corruptions(Corruption, noise_corruption, 
                ['uniform_noise', 'gaussian_noise', 'background_noise', 'impulse_noise', 
                 'uniform_noise_radial', 'gaussian_noise_radial', 'impulse_noise_radial',
                 'upsampling'], 
                 noise_template)
