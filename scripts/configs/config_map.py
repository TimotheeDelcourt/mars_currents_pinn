
# # dashboard
all_maps = 0
wind_map = 1
only_B = 0
neuron = 0


map_config = {
    # 'model_name' : 'PINN_MSO_ensemble_models_1to3_150km_fibonacci_lowaltsubset',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to30_150km_fibonacci_summer',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to30_150km_fibonacci_winter',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to31_150km_fibonacci_automn',
    'model_name' : 'PINN_MSO_ensemble_models_1to30_150km_fibonacci_winter_spring',
    # 'model_name' : 'PINN_MSO_model1_150km_fibonacci_summer_neuronoutput',
    # 'model_name' : 'PINN_MSO_ensemble_models_1to25_150km_fibonacci_low_crustal_field_regions',
    # 'model_name' : 'crustal_field_rotation_avg_winter_150km_fibonacci', # 'spring', 'autumn', 'summer', 'winter'
    'shading': False, # True or False
    'proj':'central', # north,south, or central
    'filetype': '.png',
    'frame':'spherical', # spherical or cartesian
    'colorbar_fixed':0,
    }



