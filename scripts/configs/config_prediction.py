

# # dashboard
predict_single_model = 0
predict_ensemble = 1



prediction_config = {
    'input_type': 'profile', # 'fibonacci' or 'profile'
    'num_samples': 900000, # fibonnaci_sphere
    'num_workers': 8,
    'minibatch': 1, 
    'batch_size': 50000, # only used if minibatch is 1

    # options:
   
    # if predict_single_model = 1
    'model_nb':49, # predict_single_model = 1
    'epoch_nb':None, # None = last epoch
    'reg_nb':1e11,

    # if predict_ensemble = 1
    'models_start_stop':[1,50],

    # if 'input_type': 'profile'
    'lon' : 0, # MSO longitude in degrees

    # if 'input_type': 'fibonacci_sphere'
    'alt': 150,

}