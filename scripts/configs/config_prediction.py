

# # dashboard
# predict_single_model = 1
# predict_ensemble = 0



prediction_config = {
    'input_type': 'fibonacci_sphere', #'grid', # 'fibonacci_sphere' or 'data_input'
    'num_samples': 900000, # fibonnaci_sphere
    'bootstrap_max': 100, # if predict_ensemble = 1
    'bootstrap_nb':None, # predict_single_model = 1
    'model_nb':5,
    'epoch_nb':None, # None = last epoch
    'alt': 150, # only used if input_type is 'fibonacci_sphere', Gao: 150 or 250 km
    'num_workers': 8,
    'minibatch': 1, 
    'batch_size': 50000, # only used if minibatch is 1
}