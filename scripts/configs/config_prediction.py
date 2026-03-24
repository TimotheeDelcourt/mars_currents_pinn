

# # dashboard
predict_single_model = 0
predict_ensemble = 1
predict_single_neuron = 0 # will use arguments of predict_single_model = 1
predict_time_lapse = 0



prediction_config = {
    'input_type': 'data', # 'fibonacci', 'profile', or 'data'
    'num_samples': 900000, # fibonnaci_sphere
    'num_workers': 8,
    'minibatch': 1, 
    'batch_size': 300000, # only used if minibatch is 1

    # 'models_dir': None, # may leave as None if input_type = data (season below will provide the path)
    'add_str': '', #
    # 'models_dir': 'summer(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'summer', # write '' if nothing to add to file name
    # 'models_dir': 'summer_autumn/PINN_ext_model_', # inside models/
    # 'add_str':'summer_autumn', # write '' if nothing to add to file name
    # 'models_dir': 'automn(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'automn', # write '' if nothing to add to file name
    # 'models_dir': 'autumn_winter/PINN_ext_model_', # inside models/
    # 'add_str':'autumn_winter', # write '' if nothing to add to file name
    # 'models_dir': 'winter(euler)/PINN_ext_model_', # inside models/
    # 'add_str':'winter', # write '' if nothing to add to file name
    # 'models_dir': 'winter_spring/PINN_ext_model_', # inside models/
    # 'add_str':'winter_spring', # write '' if nothing to add to file name
    # 'models_dir': 'spring(euler)/PINN_ext_model_', # inside models/
    # 'add_str':'spring', # write '' if nothing to add to file name
    'models_dir': 'all_year/PINN_ext_model_', # inside models/
    'add_str':'all_year', # write '' if nothing to add to file name
    # 'models_dir': 'low_crustal_field_regions(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'low_crustal_field_regions', # write '' if nothing to add to file name
    # 'models_dir': 'high_crustal_field_regions(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'high_crustal_field_regions', # write '' if nothing to add to file name
    # 'models_dir': 'summer_2HL/PINN_ext_model_', # inside models/
    # 'add_str':'summer_2HL', # write '' if nothing to add to file name


    # options:
   
    # if predict_single_model = 1 # for debugging
    'model_nb':1, # predict_single_model = 1
    'epoch_nb':None, # None = last epoch
    # 'reg_nb':1e11,

    # if predict_ensemble = 1
    'models_start_stop':[1,30],

    # if predict_time_lapse = 1
    'ls_list': [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5,11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75, 191.25, 213.75, 236.25, 258.75, 281.25, 303.75, 326.25, 348.75],

    # if 'input_type': 'profile'
    'lon' : 90, # MSO longitude in degrees
    'alt_max_profile':1500, # km

    # if 'input_type': 'fibonacci_sphere'
    'alt': 150,

    # if 'input_type': 'data'
    'season':['all_year'],
    'alt_max_data':650,
}
