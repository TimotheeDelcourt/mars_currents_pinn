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

furnsh("scripts/kernels.txt")

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
    sample = sample.drop(columns=['year','doy','hr','min','sec','decimal doy','msec','dec doy from SS file','BL PC Bx', 'BL PC By', 'BL PC Bz','PC x', 'PC y', 'PC z','SS Bx model','SS By model','SS Bz model'])
    sample['orbit number'] = sample['orbit number'].astype(int)
    return sample

def prepare_tensors():
    position = torch.tensor(pd.read_parquet('data/MAVEN_MSO_data_and_model.parquet', columns=['SS x','SS y','SS z']).values, dtype=torch.float32)
    torch.save(position, 'data/position.pt')
    del position
    a = pd.read_parquet('data/MAVEN_MSO_data_and_model.parquet', columns=['SS Bx','SS By','SS Bz']).values
    b = pd.read_parquet('data/MAVEN_MSO_data_and_model.parquet', columns=['BL SS Bx','BL SS By','BL SS Bz']).values
    # target = torch.tensor(, dtype=torch.float32)

def rotate_MBF_to_MSO(df_chunk = format_sample()):
    time_et = spice.datetime2et(df_chunk.time)
    rotation_matrices = np.stack([spice.pxform('IAU_MARS','MAVEN_MSO',t) for t in time_et]) # shape: (N, 3, 3)
    B_mbf = df_chunk[['PC Bx data','PC By data','PC Bz data']].values
    B_mso = np.einsum('nij,nj->ni', rotation_matrices, B_mbf)
    return B_mso

def parallel_rotate(n_processes=None, chunk_size=500000):
    df = pd.read_parquet('data/MAVEN_MSO_data_and_model.parquet', columns=['time','PC Bx data','PC By data','PC Bz data'])
    df = df[:2000000] # test
    # print(df.memory_usage(index=True).sum()) : 7 Gb
    if n_processes is None:
        n_processes = 7
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    with Pool(n_processes) as pool:
        results = pool.map(rotate_MBF_to_MSO, chunks)
    B_mso = np.concatenate(results)
    

    


if __name__ == "__main__":

    # dashboard
    perform_merge_all_dat_to_parquet = 0
    test_format = 1
    test_rotation = 0
    peform_parallel_rot = 0


    # execution
    if test_format:
        df = format_sample()
        print(df.columns)

    if perform_merge_all_dat_to_parquet:
        path = '../../Data/MAVEN_MAG/MSO_AM_BL/'
        all_mso_files = glob.glob(path+'*.dat')
        all_mso_files = tqdm(all_mso_files)
        for i, f in enumerate(all_mso_files):
            df = format_sample(f)
            if i == 0:
                df.to_parquet('data/MAVEN_MSO_data_and_model.parquet', engine = 'fastparquet', compression='zstd')
            else:
                df.to_parquet('data/MAVEN_MSO_data_and_model.parquet', engine = 'fastparquet', compression='zstd', append=True)

    if test_rotation:
        time_start = time.time()
        rotate_MBF_to_MSO()
        time_stop = time.time()
        print(f'elapsed time : {time_stop-time_start} seconds')
        print(f'Estimated time for 3500 files : {(time_stop-time_start)*3500/3600} hours')

    if peform_parallel_rot:
        parallel_rotate()


   
