import numpy as np
from pypcd4 import PointCloud
import click
import pathlib
import multiprocess
from corruption import Corruption

ASubsetPath = {
    'ColumbiaPark': "ColumbiaPark/2018-10-11"
}

@click.command()
@click.option('--corruption', help='corruption type', default='rain')
@click.option('--severity', help='corruption severity, from 1 to 5', default=-1)
@click.option('--dataset', help='dataset name, kitti or apollo', default='kitti')
@click.option('--subset', help='sequence number of kitti odometry, or subset of apollo southbay')
@click.option('--dataroot', help='root folder of the dataset')
@click.option('--nworkers', help='number of workers for parallelizing the permutation', default=8)
def config(corruption, severity, dataset, subset, dataroot, nworkers):
    cfg = {
        'corruption': corruption,
        'severity': int(severity),
        'dataset': dataset,
        'subset': subset,
        'dataroot': dataroot,
        'nworkers': int(nworkers)
    }
    return cfg
    
def kitti_handler(inpath, outpath, cfg):
    """
    read, augment and save point clouds
    """
    if (outpath / inpath.name).exists():
        return
    scan = np.fromfile(inpath, dtype=np.float32).reshape(-1, 4)
    aug_scan = getattr(Corruption, cfg['corruption'])(scan, cfg['severity'])
    aug_scan[:, :4].astype(np.float32).tofile(outpath / inpath.name)

def apollo_handler(inpath, outpath, cfg):
    """
    read, augment and save point clouds
    Note: timestamp will be lost
    """
    if (outpath / inpath.name).exists():
        return
    pcd = PointCloud.from_path(inpath)
    scan = pcd.numpy(("x", "y", "z", "intensity"))
    timestamps = pcd.numpy(("timestamp",))
    scan[:, 3] /= 255
    aug_scan = getattr(Corruption, cfg['corruption'])(scan, cfg['severity'])
    aug_scan = aug_scan[:, :4]
    aug_scan[:, 3] *= 255
    aug_pcd = PointCloud.from_points(aug_scan, pcd.fields[:-1], pcd.types[:-1])
    aug_pcd.save(outpath / inpath.name)

def main(cfg):
    dataroot = pathlib.Path(cfg['dataroot'])
    if cfg['dataset'] == 'kitti':
        inpath = dataroot / f'kitti/dataset/sequences/{cfg["subset"]}/velodyne'
        outpath = dataroot / f'kitti_{cfg["corruption"]}_{cfg["severity"]}/dataset/sequences/{cfg["subset"]}/velodyne'
        outpath.mkdir(parents=True, exist_ok=True)
        filepaths = inpath.glob('*.bin')
        handler = kitti_handler
    elif cfg['dataset'] == 'kitti_raw':
        inpath = dataroot / f'{cfg["subset"]}/velodyne_points/bins'
        outpath = dataroot / f'{cfg["subset"]}/velodyne_points/{cfg["corruption"]}_{cfg["severity"]}/'
        outpath.mkdir(parents=True, exist_ok=True)
        filepaths = inpath.glob('*.bin')
        handler = kitti_handler
    elif cfg['dataset'] == 'apollo':
        inpath = dataroot / f'apollo/TestData/{ASubsetPath[cfg["subset"]]}/pcds'
        outpath = dataroot / f'apollo_{cfg["corruption"]}_{cfg["severity"]}/TestData/{ASubsetPath[cfg["subset"]]}/pcds'
        outpath.mkdir(parents=True, exist_ok=True)
        filepaths = inpath.glob('*.pcd')
        handler = apollo_handler
    else:
        raise "Unsupported dataset."
    with multiprocess.Pool(cfg['nworkers']) as pool:
        pool.map(lambda x: handler(x, outpath, cfg), filepaths)

if __name__ == '__main__':
    cfg = config(standalone_mode=False)
    main(cfg)
