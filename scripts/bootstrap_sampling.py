import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def prepare_bootstrap_dataloaders(input,target,nb_all, batch_size, n_cpus, replacement = True):
    
    # Sample orbit -> return matching data indices -----------------------------
    nb_unique = torch.unique(nb_all)
    
    if replacement:
        print('Preparing bootstrap with replacement.')
        nb_unique_train_indices = torch.randint_like(input=nb_unique,high=len(nb_unique),dtype=int)
        nb_train = nb_unique[nb_unique_train_indices]
        print('Unique orbits / total orbits: ', len(torch.unique(nb_train))/len(nb_train)) #check
    else:
        print('Preparing bootstrap without replacement.')
        data_split = 0.3
        nb_train, _ = train_test_split(nb_unique.numpy(), test_size=data_split)
        nb_train = torch.tensor(nb_train, dtype=int)
        print('Sampled orbits / total orbits: ', len(nb_train)/len(nb_unique)) #check


    train_mask = torch.isin(nb_all, nb_train)
    train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    
    val_mask = ~train_mask  # Invert mask for validation
    val_indices = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)


    # Retrieve corresponding data ----------------------------------
    train_input, train_target = input[train_indices], target[train_indices]
    test_input, test_target = input[val_indices], target[val_indices]

    # # testing rest of the code without sampling ---------------------------------------------
    # # all lines above must be commented !
    # train_input = input.clone()
    # train_target = target.clone()
    # first_40_percent_index = int(0.4*len(train_input))
    # test_input = input[:first_40_percent_index].clone()
    # test_target = target[:first_40_percent_index].clone()
    # print('Fake sampling done!')
    # # ---------------------------------------------------------------------------------------

    # Prepare dataloaders ------------------------------------------
    train_dataset = TensorDataset(train_input, train_target)
    val_dataset = TensorDataset(test_input, test_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus)
    val_loader = DataLoader(val_dataset,  shuffle=True, batch_size=batch_size, num_workers=n_cpus)
    
    print(f'Bootstrap prepared!')

    return train_loader, val_loader


# def prepare_bootstrap_dataloaders_no_orbits(input,target, batch_size, n_cpus, MGS_input, MGS_target, percent_MGS):
    
#     ''' Only samples data points.'''

#     print('Preparing bootstrap...')

#     # Sample orbit -> return matching data indices -----------------------------
#     n = len(input)
    
#     all_indices = torch.arange(n)
#     train_indices = torch.randint(high=n,size=(n,))
#     val_indices_bool = torch.isin(elements=all_indices, test_elements=train_indices, invert=True)
#     val_indices = torch.nonzero(val_indices_bool).squeeze()
   

#     print(len(torch.unique(train_indices))/len(train_indices)) #check

  
#     # Retrieve corresponding data ----------------------------------
#     train_input, train_target = input[train_indices], target[train_indices]
#     val_input, val_target = input[val_indices], target[val_indices]

#     # downsample MGS data ---------------------------------------
#     MGS_size_new = int(n*percent_MGS)
#     MGS_indices_new = torch.randperm(len(MGS_input))[:MGS_size_new]
#     MGS_input = MGS_input[MGS_indices_new]
#     MGS_target = MGS_target[MGS_indices_new]

#     train_input = torch.cat((train_input, MGS_input),0)
#     train_target = torch.cat((train_target, MGS_target),0)

#     print('Amount of MAV & MGS data:', input.shape[0], MGS_input.shape[0])

  
#     # Prepare dataloaders ------------------------------------------
#     train_dataset = TensorDataset(train_input, train_target)
#     val_dataset = TensorDataset(val_input, val_target)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus)
#     val_loader = DataLoader(val_dataset,  shuffle=True, batch_size=batch_size, num_workers=n_cpus)
    
#     print(f'Bootstrap prepared!')

#     return train_loader, val_loader

# def split_wrt_orbits(input,target,nb_all, batch_size, n_cpus):
    
#     print('Preparing bootstrap...')

#     # Sample orbit -> return matching data indices -----------------------------
#     nb_unique = torch.unique(nb_all)
#     nb_train, nb_val = train_test_split(nb_unique.numpy(), test_size=0.3)

#     train_input, train_target = torch.empty((0,3)), torch.empty((0,3))
#     test_input, test_target = torch.empty((0,3)), torch.empty((0,3))

#     # i = 0
#     # for input_i, target_i, orbit_i in zip(input, target, nb_all):
#     #     if orbit_i in nb_train:
#     #         train_input = torch.vstack((train_input, input_i))
#     #         train_target = torch.vstack((train_target, target_i))
#     #     elif orbit_i in nb_val:
#     #         test_input = torch.vstack((test_input, input_i))
#     #         test_target = torch.vstack((test_target, target_i))
#     #     progress = i * 100 / len(input)
#     #     if progress % 10 < 0.1:
#     #         print(f"{progress:.0f} %", end='\r', flush=True)
#     #     i += 1

#     train_indices = []
#     progress_old = -1
#     for i,nb in enumerate(nb_train):
#         cond = torch.isin(elements=nb_all,test_element=nb)
#         nb_indices = torch.nonzero(cond).squeeze().tolist()
#         if isinstance(nb_indices,int):
#             train_indices.append(nb_indices)
#         else:
#             train_indices.extend(nb_indices)
#         progress = i * 100 / len(nb_train)
#         if abs(progress - round(progress)) < 1e-1:
#             if round(progress) != round(progress_old):
#                 print(f"{progress:.0f} %")#, end='\r', flush=True)
#             progress_old = progress

#     train_indices = torch.tensor(train_indices, dtype=int)
#     val_indices_bool = torch.isin(elements=nb_all, test_elements=nb_train, invert=True)
#     val_indices = torch.nonzero(val_indices_bool).squeeze()

    

#     # Prepare dataloaders ------------------------------------------
#     train_dataset = TensorDataset(train_input, train_target)
#     val_dataset = TensorDataset(test_input, test_target)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus)
#     val_loader = DataLoader(val_dataset,  shuffle=True, batch_size=batch_size, num_workers=n_cpus)
    
#     print(f'Bootstrap prepared!')

#     return train_loader, val_loader