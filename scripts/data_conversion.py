import pandas as pd
import numpy as np
import glob
from tqdm import trange, tqdm
import torch
import warnings
warnings.filterwarnings("ignore")
from spiceypy import *
import spiceypy as spice
import time
from multiprocessing import Pool, cpu_count
# from . import utils
import utils



def format_sample(dat_file = '../../Data/MAVEN_MAG/MSO_AM_BL/2014283pc.dat'):
    sample = pd.read_csv(dat_file, delim_whitespace=True, header=None)
    sample.columns = ['year','doy','hr','min','sec','msec','decimal doy',
                        'PC x','PC y','PC z','PC Bx data','PC By data','PC Bz data',
                        'lon','lat','alt','dec doy from SS file', 'SS x',
                        'SS y','SS z','SS lon','local time','SS Bx',
                        'SS By','SS Bz','SS lat','SS Bx model','SS By model',
                        'SS Bz model','BL SS Bx','BL SS By','BL SS Bz','BL PC Bx',
                        'BL PC By','BL PC Bz','orbit number']
    sample.insert( 0 , 'time', pd.to_datetime(sample[['year', 'doy', 'hr', 'min', 'sec']].astype(int).astype(str).agg('-'.join, axis=1), format='%Y-%j-%H-%M-%S')    )
    sample = sample.drop(columns=['year','doy','hr','min','sec','decimal doy','msec','dec doy from SS file','SS Bx model','SS By model','SS Bz model',
                                  ])#'BL PC Bx','BL PC By','BL PC Bz'])
    sample['orbit number'] = sample['orbit number'].astype(int)
    return sample

def merge_all_dat_to_parquet():
    path = '../../Data/MAVEN_MAG/MSO_AM_BL/'
    all_mso_files = glob.glob(path+'*.dat')
    all_mso_files = tqdm(all_mso_files)
    for i, f in enumerate(all_mso_files):
        df = format_sample(f)
        if i == 0:
            df.to_parquet('data/MAVEN_MSO_data.parquet', engine = 'fastparquet', compression='zstd')
        else:
            df.to_parquet('data/MAVEN_MSO_data.parquet', engine = 'fastparquet', compression='zstd', append=True)

  

def rotate_MBF_to_MSO(df_chunk = format_sample()):
    furnsh("scripts/kernels.txt")
    time_et = spice.datetime2et(df_chunk.time)
    rotation_matrices = np.stack([spice.pxform('IAU_MARS','MAVEN_MSO',t) for t in time_et]) # shape: (N, 3, 3)
    # MEMORY OVERLOADED! rotation_matrices = spice.cyice.cyice.pxform_v('IAU_MARS', 'MAVEN_MSO', time_et)  # shape: (N,3,3)
    B_mbf = df_chunk[['Br','Bt','Bp']].values
    B_mso = np.einsum('nij,nj->ni', rotation_matrices, B_mbf)
    return B_mso

def parallel_rotate(n_processes=None, chunk_size=1000000):
    df = pd.read_parquet('data/MAVEN_MSO_data.parquet', columns=['time']) # first add PC_crust columns
    crustal_field_pc = torch.load('data/crustal_field_pc_xyz.pt')
    assert len(crustal_field_pc) == len(df), "crustal_field_pc.pt has a different length than the dataframe"
    df['Br'] = crustal_field_pc[:,0].numpy()
    df['Bt'] = crustal_field_pc[:,1].numpy()
    df['Bp'] = crustal_field_pc[:,2].numpy()
    # print(df.memory_usage(index=True).sum()) # 6 Gb

    if n_processes is None:
        n_processes = 7
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    with Pool(n_processes) as pool:
        results = pool.map(rotate_MBF_to_MSO, chunks)
    B_mso = np.concatenate(results)
    B_mso = torch.tensor(B_mso, dtype=torch.float32)
    torch.save(B_mso,'data/crustal_field_mso.pt')

def pc_sph2cart(position_pc=torch.load('data/position_pc.pt'), field_pc=torch.load('data/crustal_field_pc.pt')):
    position_pc[:,1] = 90-position_pc[:,1] # colat to lat
    position_pc[:,1] = torch.deg2rad(position_pc[:,1])
    position_pc[:,2] = torch.deg2rad(position_pc[:,2])
    field_pc_xyz = utils.field_spher_to_cart(position_pc,field_pc)
    torch.save(field_pc_xyz, 'data/crustal_field_pc_xyz.pt')

def make_subsolarlongitude_series(): # in MBF frame
    df = pd.read_parquet('data/MAVEN_MSO_data.parquet', columns=['time'])
    # df = df[::10000].copy()
    subsolar_xyz = np.zeros((len(df),3))
    furnsh("scripts/kernels.txt")
    for i,t in enumerate(tqdm(df.time)):
        et = spice.datetime2et(t)
        sun_pos, _, _ = spice.subslr(method='NEAR POINT/ELLIPSOID', target='Mars', et=et, fixref='IAU_MARS', abcorr='NONE', obsrvr='Mars')
        subsolar_xyz[i] = sun_pos
        
    subsolar_xyz = torch.tensor(subsolar_xyz, dtype=torch.float32)
    _, colat_rad, lon_rad = utils.cartesian_to_spherical(subsolar_xyz[:,0], subsolar_xyz[:,1], subsolar_xyz[:,2])
    del subsolar_xyz
    subsolar_lat_lon = torch.vstack((90 - torch.rad2deg(colat_rad), torch.rad2deg(lon_rad))).T
    print(subsolar_lat_lon)
    print(f'lat min, max: {subsolar_lat_lon[:,0].min()}, {subsolar_lat_lon[:,0].max()}')
    print(f'lon min, max: {subsolar_lat_lon[:,1].min()}, {subsolar_lat_lon[:,1].max()}')
   
    torch.save(subsolar_lat_lon, 'data/subsolar_lat_lon.pt')

# def make_subsolarlongitude_series_parallel():
#     df = pd.read_parquet('data/MAVEN_MSO_data.parquet', columns=['time'])
#     df = df[:100].copy()
#     # subsolar_xyz = np.zeros((len(df),3))
#     furnsh("scripts/kernels.txt")
#     def compute_subslr(t):
#         et = spice.datetime2et(t)
#         sun_pos, _, _ = spice.subslr(method='NEAR POINT/ELLIPSOID', target='Mars', et=et, fixref='IAU_MARS', abcorr='NONE', obsrvr='Mars')
#         return sun_pos
    
#     time_chunks = np.array_split(df.time, 7)
#     time_chunks = [tc.tolist() for tc in time_chunks]
#     del df
#     with Pool(7) as pool:
#         results = pool.map(compute_subslr, time_chunks)
#     subsolar_xyz = np.array(results)
#     subsolar_xyz = torch.tensor(subsolar_xyz, dtype=torch.float32)
#     _, colat_rad, lon_rad = utils.cartesian_to_spherical(subsolar_xyz[:,0], subsolar_xyz[:,1], subsolar_xyz[:,2])
#     del subsolar_xyz
#     subsolar_lat_lon = torch.vstack((90 - torch.rad2deg(colat_rad), torch.rad2deg(lon_rad))).T
#     print(subsolar_lat_lon)
    # print(f'lat min, max: {subsolar_lat_lon[:,0].min()}, {subsolar_lat_lon[:,0].max()}')
    # print(f'lon min, max: {subsolar_lat_lon[:,1].min()}, {subsolar_lat_lon[:,1].max()}')
   
    # torch.save(subsolar_lat_lon, 'data/subsolar_lat_lon.pt')


if __name__ == "__main__":

    # dashboard
    perform_merge_all_dat_to_parquet = 0
    test_format = 0
    test_rotation = 0
    perform_parallel_rot = 0
    perform_subsolar_longitude_series = 1


    # execution
    if test_format:
        df = format_sample()
        print(df.head())


    if perform_merge_all_dat_to_parquet:
        merge_all_dat_to_parquet()

    if test_rotation:
        time_start = time.time()
        rotate_MBF_to_MSO()
        time_stop = time.time()
        print(f'elapsed time : {time_stop-time_start} seconds')
        print(f'Estimated time for 3500 files : {(time_stop-time_start)*3500/3600} hours')

    if perform_parallel_rot:
        parallel_rotate()

    if perform_subsolar_longitude_series:
        make_subsolarlongitude_series()
        # make_subsolarlongitude_series_parallel()
