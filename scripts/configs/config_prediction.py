

# # dashboard
predict_single_model = 0
predict_ensemble = 1



prediction_config = {
    'input_type': 'data', # 'fibonacci', 'profile', or 'data'
    'num_samples': 900000, # fibonnaci_sphere
    'num_workers': 8,
    'minibatch': 1, 
    'batch_size': 300000, # only used if minibatch is 1
    # 'models_dir': 'automn(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'automn', # write '' if nothing to add to file name
    # 'models_dir': 'summer(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'summer', # write '' if nothing to add to file name
    'models_dir': 'winter(euler)/PINN_ext_model_', # inside models/
    'add_str':'winter', # write '' if nothing to add to file name
    # 'models_dir': 'spring(euler)/PINN_ext_model_', # inside models/
    # 'add_str':'spring', # write '' if nothing to add to file name
    # 'models_dir': 'low_crustal_field_regions(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'low_crustal_field_regions', # write '' if nothing to add to file name
    # 'models_dir': 'high_crustal_field_regions(sisma)/PINN_ext_model_', # inside models/
    # 'add_str':'high_crustal_field_regions', # write '' if nothing to add to file name
    # 'models_dir': 'summer_autumn/PINN_ext_model_', # inside models/
    # 'add_str':'summer_autumn', # write '' if nothing to add to file name

    

    # options:
   
    # if predict_single_model = 1 # for debugging
    'model_nb':28, # predict_single_model = 1
    'epoch_nb':None, # None = last epoch
    # 'reg_nb':1e11,

    # if predict_ensemble = 1
    'models_start_stop':[1,35],

    # if 'input_type': 'profile'
    'lon' : 90, # MSO longitude in degrees
    'alt_max_profile':1500, # km

    # if 'input_type': 'fibonacci_sphere'
    'alt': 150,

    # if 'input_type': 'data'
    'season':'winter',
    'alt_max_data':500,
}