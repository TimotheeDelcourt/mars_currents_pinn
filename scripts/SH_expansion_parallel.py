import pyshtools as sh
import numpy as np
from multiprocessing import Pool, cpu_count
import torch


coeffs = sh.SHMagCoeffs.from_file('crustal_field_model/PINN2025.sh', normalization='schmidt')

# Expansion ---
def expand_chunk(points):
    alt, lat, lon = points[:,0], points[:,1], points[:,2]
    expanded_chunk = coeffs.expand(alt = alt, lat = lat, lon = lon, degrees=True)
    return expanded_chunk


if __name__=='__main__':


    points = torch.load('data/position_pc.pt').numpy()
    alt, lat, lon = points[:,0], points[:,1], points[:,2]
    r = alt + 3393.5

    num_cpus = cpu_count() - 1
    points = np.stack((r, lat, lon), axis=1)
    points = points.squeeze()
    points_chunked = np.array_split(points, num_cpus)
    
    
    with Pool(num_cpus) as pool:
        expansion = pool.map(expand_chunk, points_chunked)

    expansion = np.concatenate(expansion, axis=0)
    expansion = torch.tensor(expansion, dtype=torch.float32)
    torch.save(expansion, 'data/crustal_field_pc.pt')

    
    

