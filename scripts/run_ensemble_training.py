import configs.config_training as config
import os
import torch
import bootstrap_sampling
import shutil
from torch import optim
from training import train, train_noval
from neuralnets import NeuralNet_indep
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
# import utils.utils as utils
import utils.npy2json_sisma as npy2json
import json

while (os.path.basename(os.getcwd()) != 'mars_currents_pinn'):
    os.chdir('../')
    


def run_ensemble_training():

    # 2026/05/01 update, only consider Ls values from now on, no "seasons"
    # seasons = config.training_config['season_filter']
    # if not isinstance(seasons, list):
    #     seasons = [seasons]
    ls_list = config.training_config['ls_list']

    for ls in ls_list:
        print(f'LS: {ls} degrees')

        # Bootstrap iteration---------------------------------------------
        for _ in range(config.training_config['ensemble_size']):
        # for smoothness_lambda in [1e6,1e7,1e8]:
        # for smoothness_lambda in [1e9,1e10,1e11]:

            print('Making folder')

            # # smoothing lambda test -----------
            # folder_name = 'models/PINN_ext_smoothness_reg_'+f'{smoothness_lambda:.0e}'
            # os.makedirs(folder_name)
            # print(f'Training model with smoothness lambda {smoothness_lambda:.0e}...')
            # # (comment make folder below) -----
            

            # Make folder-------------------------------------------------
            counter = config.training_config['bootstrap_counter_start']
            add_str  = config.training_config['add_folder_str']
            base_folder_name = f'models/ls{ls}'+add_str+'/PINN_ext_model_'
            # Keep creating new folders with incremented names until one with a unique name is found
            while True:
                folder_name = base_folder_name+str(counter)
                # Check if the folder exists
                if not os.path.exists(folder_name):
                    # If the folder does not exist, create it
                    os.makedirs(folder_name)
                    break
                else:
                    # If the folder exists, increment the counter and try again
                    counter += 1
            print(f'Training model {counter}...')

            # Save the current config file in the folder----------------
            
            # and training script
            shutil.copyfile('scripts/training.py', folder_name+'/training.py')
            shutil.copyfile('scripts/neuralnets.py', folder_name+'/neuralnets.py')
            shutil.copyfile('scripts/run_ensemble_training.py', folder_name+'/run_ensemble_training.py')
            shutil.copyfile('scripts/configs/config_training.py', folder_name+'/config_training.py')
            os.makedirs(folder_name+'/models/')

            # Load and sample datasets ----------------------------------
            print('Loading data')
            input_xyz = torch.load('data/position_mso.pt')
            input_sph = torch.load('data/position_mso_spherical.pt')
            # alt = input_sph[:,0]
            
            
            
            crustal_field_mso = torch.load('data/crustal_field_mso.pt')
            observation_mso = torch.load('data/observation_mso.pt')
            target = observation_mso - crustal_field_mso

            if config.training_config['random_parameters'] == False:
                num_neurons_per_layer=config.training_config['num_neurons_per_layer']
                alt_max = config.training_config['altitude_max']
                # l1_lambda = config.training_config['l1_lambda']
                include_alt = config.training_config['include_alt']
                lim = 60
            elif config.training_config['random_parameters'] == True:
                include_alt = False #np.random.choice([True,False])
                alt_max_list = config.training_config['altitudes_max']
                if len(alt_max_list) == 1:
                    alt_max = alt_max_list[0]
                else:
                    alt_max = np.random.randint(alt_max_list[0], alt_max_list[1]+1)
                # l1_lambda = np.random.choice(config.training_config['l1_lambdas'])
                num_neurons_per_layer = np.random.randint(config.training_config['nums_neurons_per_layer'][0], config.training_config['nums_neurons_per_layer'][1]+1)
                lim = np.random.choice(config.training_config['crop_outlier'])

            smoothness_lambda = 1e10
            l1_lambda = 0
            print(f'Altitude max: {alt_max} km')
            print(f'Number of neurons per layer: {num_neurons_per_layer}')
            # print(f'L1 lambda: {l1_lambda}')
            print(f'Include altitude: {include_alt}')
            print(f'Smoothness lambda: {smoothness_lambda}')
            print(f'Crop limit: {lim} nT')


            condition = (input_sph[:,0] <= alt_max) & torch.all((target <= lim) & (target >= -lim), dim=1)

            # seasonal constraint----------------------------------------------------------------------
            # 2026/05/01 update, only consider Ls values from now on, no "seasons":
            target_ls = ls
            # commented out block below ------------
            # if config.training_config['season_filter'] is not None:
            #     if season == 'summer':
            #         target_ls = 90
            #     elif season == 'winter':
            #         target_ls = 270
            #     elif season == 'spring':
            #         target_ls = 0
            #     elif season  == 'autumn':
            #         target_ls = 180
            #     elif season == 'summer_autumn':
            #         target_ls = 135
            #     elif season == 'autumn_winter':
            #         target_ls = 225
            #     elif season == 'winter_spring':
            #         target_ls = 315
            #     elif season == 'spring_summer':
            #         target_ls = 45
            #     else:
            #         raise ValueError('season_filter must be "summer", "winter", "spring", "autumn" or None')
            # shifted left below --------------------
            angle_half_band = config.training_config['ls_angle_band']/2
            # print(f'Applying seasonal filter: {config.training_config["season_filter"]} with Ls band of ±{angle_half_band} degrees around Ls={target_ls} degrees')
            ls = torch.load('data/Ls_series.pt')
            lower_bound = target_ls - angle_half_band
            upper_bound = target_ls + angle_half_band
            lower_bound = lower_bound % 360
            upper_bound = upper_bound % 360
            if lower_bound > upper_bound:
                condition2 = (ls >= lower_bound) | (ls <= upper_bound)
            else:
                condition2 = (ls <= upper_bound) & (ls >= lower_bound)
            condition = condition & condition2
            # -----------------------------------------------------------------------------------------

            # crustal field condition------------------------------------------------------------------
            if config.training_config['curstal_field_condition'] is not None:
                limit = config.training_config['crustal_field_limit']
                total_field = torch.sqrt(crustal_field_mso[:,0]**2 + crustal_field_mso[:,1]**2 + crustal_field_mso[:,2]**2)
                if config.training_config['curstal_field_condition'] == 'low':
                    condition3 = total_field <= limit
                    print(f'Applying crustal field condition: low crustal field regions with total crustal field ≤ {limit} nT')
                elif config.training_config['curstal_field_condition'] == 'high':
                    condition3 = total_field >= limit
                    print(f'Applying crustal field condition: high crustal field regions with total crustal field ≥ {limit} nT')
                else:
                    raise ValueError('curstal_field_condition must be "low", "high" or None')
                condition = condition & condition3
            # -----------------------------------------------------------------------------------------


            target = target[condition]

            if include_alt == True:
                input = torch.concatenate((input_xyz, input_sph[:,0].unsqueeze(1)), dim=1)
                input = input[condition]
                num_inputs = 4
                alt_mean = torch.mean(input[:, 3]).item()
                alt_std = torch.std(input[:, 3]).item()
                print('alt_mean, alt_std: ', alt_mean, alt_std)
            else:
                input = input_xyz
                input = input[condition]
                num_inputs = 3
                alt_mean = None
                alt_std = None


            xyz_mean = 0#torch.mean(input[:, :3]).item()
            xyz_std = torch.std(input[:, :3]).item()
            
            model_params = {
                'xyz_mean': xyz_mean,
                'xyz_std': xyz_std,
                'alt_mean': alt_mean,
                'alt_std': alt_std,
                'num_inputs': num_inputs,
                'l1_lambda': l1_lambda,
                'num_neurons_per_layer': num_neurons_per_layer,
                'alt_max': alt_max,
                'crop_limit': lim,
                'smoothness_lambda': smoothness_lambda,
                'include_alt': include_alt,
            }
            np.save(folder_name+'/model_params.npy', model_params)

            model_params_json = npy2json.convert(model_params)
            with open(folder_name+'/model_params.json', 'w') as f:
                # f.write(model_params_json)
                json.dump(model_params_json, f)
            

            # std_params = (xyz_mean, xyz_std, alt_mean, alt_std)
            # torch.save(std_params, folder_name+'/std_params.pt')
            print('Input shape: ', input.shape)
            # print('xyz_mean, xyz_std: ', xyz_mean, xyz_std)
            # print('alt_mean, alt_std: ', alt_mean, alt_std)

            # Device ---------------------------------------------------
            if torch.cuda.is_available():
                DEVICE = torch.device('cuda')
            else:
                DEVICE = 'cpu'
            print(f'''Device: {DEVICE}''')

            # Load network ---------------------------------------------

            model = NeuralNet_indep(
                num_hidden_layers=config.training_config['num_hidden_layers'],
                num_neurons_per_layer=num_neurons_per_layer,
                xyz_mean=xyz_mean,
                xyz_std=xyz_std,
                num_inputs=num_inputs,
                alt_mean=alt_mean,
                alt_std=alt_std,
                activation=config.training_config['activation']
            ).to(DEVICE)

            # parameters -------------------------------------------------
            num_epochs = config.training_config['num_epochs']
            optimizer = optim.Adam(model.parameters(), lr=config.training_config['learning_rate'])
            n_cpus = config.training_config['n_cpus']
            lossfn = config.training_config['lossfn']
            batch_size = config.training_config['batch_size']

            if config.training_config['validation']:
                orbit_nb = torch.load('data/orbit_nb.pt')
                orbit_nb = orbit_nb[condition]
                train_loader, val_loader = bootstrap_sampling.prepare_bootstrap_dataloaders(input, target, orbit_nb, 
                                                                                        batch_size,
                                                                                        n_cpus,
                                                                                        replacement=config.training_config['sample_with_replacement'],
                                                                                        )
                
                # assert 1 == 0, "Debugging stop"
                train(model,train_loader,val_loader, num_epochs, optimizer, DEVICE,
                    folder_name, n_cpus, lossfn, l1_lambda=l1_lambda, smoothness_lambda=smoothness_lambda)
            else:
                train_dataset = TensorDataset(input, target)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus)

                train_noval(model, train_loader, num_epochs, optimizer, DEVICE,
                            folder_name, n_cpus, lossfn, l1_lambda = config.training_config['l1_lambda'])
                
            print('')
            del model

if __name__ == "__main__":
    
    run_ensemble_training()


        



