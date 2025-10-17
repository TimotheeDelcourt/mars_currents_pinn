

# # dashboard
predict_single_model = 0
predict_ensemble = 1



prediction_config = {
    'input_type': 'fibonacci_sphere', #'grid', # 'fibonacci_sphere' or 'data_input'
    'num_samples': 900000, # fibonnaci_sphere
    'alt': 150, # only used if input_type is 'fibonacci_sphere', Gao: 150 or 250 km
    'num_workers': 8,
    'minibatch': 1, 
    'batch_size': 50000, # only used if minibatch is 1
   
    # predict_single_model = 1
    'model_nb':69, # predict_single_model = 1
    'epoch_nb':None, # None = last epoch

    # predict_ensemble = 1
    'models_start_stop':[32,66],
}