

# # dashboard
predict_single_model = 0
predict_ensemble = 1



prediction_config = {
    'input_type': 'fibonacci', # 'fibonacci' or 'profile'
    'num_samples': 900000, # fibonnaci_sphere
    'num_workers': 8,
    'minibatch': 1, 
    'batch_size': 100000, # only used if minibatch is 1
    'models_dir': 'MBFaligned/PINN_ext_model_', # inside models/
    'add_str':'MBFaligned_wrong', # write '' if nothing to add to file name

    # options:
   
    # if predict_single_model = 1
    'model_nb':3, # predict_single_model = 1
    'epoch_nb':None, # None = last epoch
    # 'reg_nb':1e11,

    # if predict_ensemble = 1
    'models_start_stop':[1,13],

    # if 'input_type': 'profile'
    'lon' : 90, # MSO longitude in degrees
    'alt_max':1500, # km

    # if 'input_type': 'fibonacci_sphere'
    'alt': 150,

}