import pyshtools as sh
import numpy as np
import pandas as pd
from multiprocessing import Pool


import maps

def deg2km(lmax):
    radius = 3393.5
    resolution_0km = 2*np.pi*radius/lmax
    print('Resolution surface: ',resolution_0km)

# deg2km(lmax = 139)



def expand_chunk(coeffs,points):
    assert isinstance(coeffs, sh.SHMagCoeffs)
    r_chunk = points[:,0]
    colat_chunk = points[:,1]
    lon_chunk = points[:,2]
    colat_chunk[colat_chunk == 0] = 1e-4  # Avoid division by zero in spherical harmonics expansion
    colat_chunk[colat_chunk == np.pi] = np.pi - 1e-4  # Avoid division by zero in spherical harmonics expansion
    expanded_chunk = coeffs.expand(colat = colat_chunk, lon = lon_chunk, degrees=False, r = r_chunk*1000)
    return expanded_chunk



def expand_chunk_wrapper(entries):
    D25 = entries[0]
    points = entries[1]
    return expand_chunk(D25, points)






if __name__=='__main__':

    season = 'winter'  # 'spring', 'autumn', None, 'summer', 'winter'

    coeffs = sh.SHMagCoeffs.from_file(f'crustal_field_model/PINN2025_rotation_avg_{season}.sh')

   
    input_reg = pd.read_csv('predictions/PINN_MSO_ensemble_models_1to11_150km_fibonacci_summer.csv')
    input_reg = input_reg[['alt','lat','lon']]
    r = np.array(input_reg['alt']) + 3393.5
    lon = np.deg2rad(input_reg['lon'])
    colat = np.deg2rad(90 - input_reg['lat'])
    points = np.stack((r,colat,lon),axis=1)

    # Execute ---
    num_cpus = 8
    
    points_chunked = np.array_split(points, num_cpus)
    entries = [[coeffs, points_chunked[i]] for i in range(num_cpus)]
    
    
    with Pool(num_cpus) as pool:
        expansion = pool.map(expand_chunk_wrapper, entries)
    expansion = np.concatenate(expansion, axis=0)  # Flatten results

    input_reg['Br'] = expansion[:,0]
    input_reg['Bt'] = expansion[:,1]
    input_reg['Bp'] = expansion[:,2]
    input_reg['B'] = np.sqrt(expansion[:,0]**2 + expansion[:,1]**2 + expansion[:,2]**2)

    
    input_reg.to_csv(f'predictions/crustal_field_rotation_avg_{season}_150km_fibonacci.csv', index=False)

