
import numpy as np

# dashboard
all_maps = 0
wind_map = 1
only_B = 0
neuron = 0
wind_map_time_lapse = 0



map_config = {
    # 'model_name' : 'PINN_MSO_ensemble_models_1to3_150km_fibonacci_lowaltsubset',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to30_150km_fibonacci_summer',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to30_150km_fibonacci_winter',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to31_150km_fibonacci_autumn',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to30_150km_fibonacci_spring',
    'model_name' : 'PINN_MSO_ensemble_models_1to30_111km_fibonacci_all_year',
    # 'model_name' : 'PINN_LT_150km_fibonacci_winter',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to25_150km_fibonacci_low_crustal_field_regions',
    # 'model_name' : 'crustal_field_rotation_avg_winter_150km_fibonacci', # 'spring', 'autumn', 'summer', 'winter'
    'shading': False, # True or False
    'proj':'central', # north,south, or central
    'filetype': '.png',
    'frame':'spherical', # spherical or cartesian
    'colorbar_fixed':1,

    # if predict_time_lapse = 1 # this renders "model_name" useless
    'ls_list': np.linspace(0,360,32,endpoint=False),

    }



