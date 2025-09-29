import configs.config_training as config
import os
import torch
import bootstrap_sampling
import shutil
from torch import optim
from training import train 
from neuralnets import NeuralNet

while (os.path.basename(os.getcwd()) != 'mars_currents_pinn'):
    os.chdir('../')
    


def run_ensemble_training():

    # Bootstrap iteration---------------------------------------------
    for _ in range(config.training_config['ensemble_size']):


        # Make folder-------------------------------------------------
        counter = config.training_config['bootstrap_counter_start']
        base_folder_name = 'models/PINN_ext_bootstrap_'
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
        
        # Save the current config file in the folder----------------
        shutil.copyfile('scripts/configs/config_training.py', folder_name+'/config_training.py')
        # and training script
        shutil.copyfile('scripts/training.py', folder_name+'/training.py')


        # Load and sample datasets ----------------------------------
        input = torch.load('data/position_mso.pt')
        alt = torch.load('data/position_pc.pt')[:,0]
        alt = alt.unsqueeze(1)
        input = torch.concatenate((input, alt), dim=1)
        crustal_field_mso = torch.load('data/crustal_field_mso.pt')
        observation_mso = torch.load('data/observation_mso.pt')
        target = observation_mso - crustal_field_mso
        del crustal_field_mso, observation_mso
        orbit_nb = torch.load('data/orbit_nb.pt')




        train_loader, val_loader = bootstrap_sampling.prepare_bootstrap_dataloaders(input, target, orbit_nb, 
                                                                                    config.training_config['batch_size'],
                                                                                    config.training_config['n_cpus'],
                                                                                    )
           
        # Device ---------------------------------------------------
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = 'cpu'
        print(f'''Device: {DEVICE}''')

        # Load network ---------------------------------------------
        model = NeuralNet().to(DEVICE)

        # Training -------------------------------------------------
        num_epochs = config.training_config['num_epochs']
        optimizer = optim.Adam(model.parameters())
        n_cpus = config.training_config['n_cpus']
        lossfn = config.training_config['lossfn']

        train(model,train_loader,val_loader, num_epochs, optimizer, DEVICE,
              folder_name, n_cpus, lossfn)


run_ensemble_training()


        



