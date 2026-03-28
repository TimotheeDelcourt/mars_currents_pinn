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


while (os.path.basename(os.getcwd()) != 'mars_currents_pinn'):
    os.chdir('../')
    


def run_ensemble_training():

    # Bootstrap iteration---------------------------------------------
    # for _ in range(config.training_config['ensemble_size']):
    # for smoothness_lambda in [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3]:
    for smoothness_lambda in [1e-2,1e-1,1,10,100,1000]:

        print('Making folder')

        # smoothing lambda test -----------
        folder_name = 'models/PINN_ext_smoothness_reg_'+str(smoothness_lambda)
        os.makedirs(folder_name)
        print(f'Training model with smoothness lambda {smoothness_lambda}...')
        # (comment make folder below) -----

        # # Make folder-------------------------------------------------
        # counter = config.training_config['bootstrap_counter_start']
        # base_folder_name = 'models/PINN_ext_model_'
        # # Keep creating new folders with incremented names until one with a unique name is found
        # while True:
        #     folder_name = base_folder_name+str(counter)
        #     # Check if the folder exists
        #     if not os.path.exists(folder_name):
        #         # If the folder does not exist, create it
        #         os.makedirs(folder_name)
        #         break
        #     else:
        #         # If the folder exists, increment the counter and try again
        #         counter += 1
        # print(f'Training model {counter}...')

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
        alt = input_sph[:,0]
        
        crustal_field_mso = torch.load('data/crustal_field_mso.pt')
        observation_mso = torch.load('data/observation_mso.pt')
        target = observation_mso - crustal_field_mso

        if config.training_config['random_parameters'] == False:
            num_neurons_per_layer=config.training_config['num_neurons_per_layer']
            alt_max = config.training_config['altitude_max']
            l1_lambda = config.training_config['l1_lambda']
            include_alt = config.training_config['include_alt']
        elif config.training_config['random_parameters'] == True:
            include_alt = np.random.choice([True,False])
            alt_max = np.random.randint(config.training_config['altitudes_max'][0], config.training_config['altitudes_max'][1]+1)
            l1_lambda = np.random.choice(config.training_config['l1_lambdas'])
            num_neurons_per_layer = np.random.randint(config.training_config['nums_neurons_per_layer'][0], config.training_config['nums_neurons_per_layer'][1]+1)

        print(f'Altitude max: {alt_max} km')
        print(f'Number of neurons per layer: {num_neurons_per_layer}')
        print(f'L1 lambda: {l1_lambda}')
        print(f'Include altitude: {include_alt}')

        condition = (alt <= alt_max) & torch.all((target <= 60) & (target >= -60), dim=1)
        target = target[condition]

        if include_alt == True:
            input = torch.concatenate((input_xyz, alt.unsqueeze(1)), dim=1)
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
        }

        np.save(folder_name+'/model_params.npy', model_params)

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


        




