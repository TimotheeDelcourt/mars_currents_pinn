import torch
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import utils
from curl_function import curl_differentiable
warnings.filterwarnings("ignore")



def train(model, training_loader, validation_loader,  
          num_epochs, optimizer, device, folder_name, 
          n_cpus, lossfn):
    

    training_loader_size = len(training_loader)
    torch.set_num_threads(n_cpus)
    train_loss_hist = []
    val_loss_hist = []
    # pbar = trange(num_epochs)
    # scheduler_bool = 0
    # epoch_scheduler = 9999999

    # initial lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-3

    # for epoch in pbar:
    for epoch in range(num_epochs):

        running_loss = 0.0
        running_val_loss = 0.0

        for (x_train, y_train) in training_loader:
            
            x_train = x_train.to(device).requires_grad_(True)
            y_train = y_train.to(device)

            # PINN:
            A_pred = model(x_train)
            y_pred = curl_differentiable(x_train, A_pred)

            loss = lossfn(y_pred, y_train)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        total_loss = running_loss/training_loader_size
        train_loss_hist.append(total_loss)


        # VALIDATION -----------------------------------------------------
        del x_train, y_train, y_pred, loss, A_pred
            # multiple batches:
        size = 0
        for (x_test, y_test) in validation_loader:
        
            x_test = x_test.to(device).requires_grad_(True)
            y_test = y_test.to(device)
            A_pred = model(x_test)
            y_pred = curl_differentiable(x_test, A_pred)


            with torch.no_grad():
                val_loss = lossfn(y_pred, y_test)
                n = len(y_test)
                running_val_loss += val_loss.item() * n
                size += n

        validation_loss = running_val_loss/size           
        val_loss_hist.append(validation_loss)#.item())
        
        np.save(folder_name+'/val_loss_hist.npy', val_loss_hist)
        np.save(folder_name+'/training_history.npy', train_loss_hist)
        torch.save(model.state_dict(), os.path.join(folder_name, f'model{epoch}.pt'))

        if epoch>0:
            fig, axs=plt.subplots(1,2, figsize=(12,4))
            xaxis = range(len(train_loss_hist))
            axs[0].plot(xaxis, train_loss_hist, label="Train loss")
            axs[1].plot(xaxis, train_loss_hist, label="Train logloss")
            axs[0].plot(xaxis, val_loss_hist, label="Validation loss")
            axs[1].plot(xaxis, val_loss_hist, label="Validation logloss")
            axs[1].set_xscale("log")
            axs[1].set_yscale("log")
            for ax in axs:
                ax.set_xlabel('Epoch')
                ax.grid(True, which="both", ls=":")
                ax.legend()
            plt.savefig(os.path.join(folder_name, 'losses.png'))
            plt.close()
            
      
        # pbar.set_postfix_str(f'''Epoch {epoch}, Loss: {total_loss}, Validation Loss: {validation_loss}, lr: {optimizer.param_groups[0]['lr']}''')#{torch.sqrt(laplacian_loss/soft_con_weight)}''')  , Laplacian Loss: {laplacian_loss}
        # print("", end="", flush=True)  # Force flushing the output

        # if epoch>300:
        #     last200 = val_loss_hist[-200:]
        #     smooth = pd.DataFrame(last200).rolling(10,center=True).mean()
        #     plt.plot(last200,color='black')
        #     plt.plot(smooth,color='blue',linewidth=2)
        #     plt.savefig(os.path.join(folder_name, 'losses_zoom.png'))
        #     plt.close()
        #     m, _ = utils.slope(val_loss_hist, 200)
           
        #     if (m >= -1.6e-6) & (scheduler_bool == 0):
        #         scheduler_bool = 1
        #         epoch_scheduler = epoch
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = 1e-4

        #     elif (epoch > epoch_scheduler + 200) & (m >= -1e-9) & (val_loss_hist[-1] < val_loss_hist[-2]) & (val_loss_hist[-1] < val_loss_hist[-3]) & (val_loss_hist[-1] < val_loss_hist[-4]):
        #         print('Plateau reached!')
        #         break


def train_noval(model, training_loader, 
          num_epochs, optimizer, device, folder_name, 
          n_cpus, lossfn):
    

    training_loader_size = len(training_loader)
    torch.set_num_threads(n_cpus)
    train_loss_hist = []

    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-3

    # for epoch in pbar:
    for epoch in range(num_epochs):

        running_loss = 0.0

        for (x_train, y_train) in training_loader:
            
            x_train = x_train.to(device).requires_grad_(True)
            y_train = y_train.to(device)

            # PINN:
            A_pred = model(x_train)
            y_pred = curl_differentiable(x_train, A_pred)

            loss = lossfn(y_pred, y_train)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        total_loss = running_loss/training_loader_size
        train_loss_hist.append(total_loss)


        # VALIDATION -----------------------------------------------------
        np.save(folder_name+'/training_history.npy', train_loss_hist)
        torch.save(model.state_dict(), os.path.join(folder_name, f'model{epoch}.pt'))

        if epoch>0:
            fig, axs=plt.subplots(1,2, figsize=(12,4))
            xaxis = range(len(train_loss_hist))
            axs[0].plot(xaxis, train_loss_hist, label="Train loss")
            axs[1].plot(xaxis, train_loss_hist, label="Train logloss")
            axs[1].set_xscale("log")
            axs[1].set_yscale("log")
            for ax in axs:
                ax.set_xlabel('Epoch')
                ax.grid(True, which="both", ls=":")
                ax.legend()
            plt.savefig(os.path.join(folder_name, 'losses.png'))
            plt.close()


