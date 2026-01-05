import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")
import scripts.utils.utils as ut
from scripts.utils.season_mask import get_condition
import cmcrameri.cm as cmc
from scripts.utils.img_data_extractor import gray_scale, img2scalar

# RMSE vs LS


def compute_seasonal_mse(season):
    condition = get_condition(season)
    observation_mso = torch.load('data/observation_mso.pt')[condition]
    crustal_field_mso = torch.load('data/crustal_field_mso.pt')[condition]
    target = observation_mso - crustal_field_mso
    B_obs = torch.sqrt(target[:,0]**2 + target[:,1]**2 + target[:,2]**2).numpy()
    del observation_mso, crustal_field_mso
    try:
        prediction_df = pd.read_csv(f'predictions/data/PINN_MSO_ensemble_models_1to30_{season}_data_500km.csv',usecols=['Br','Bt','Bp'])
    except:
        prediction_df = pd.read_csv(f'predictions/data/PINN_MSO_ensemble_models_1to31_{season}_data_500km.csv',usecols=['Br','Bt','Bp'])
    B_pred = np.sqrt(prediction_df['Br'].values**2 + prediction_df['Bt'].values**2 + prediction_df['Bp'].values**2)
    del prediction_df
    mse = np.mean((B_obs-B_pred)**2)
    print(f'MSE = {mse:.1f}')
    return mse

def compute_mse_list(save=0):
    seasons_str = ['spring','spring_summer','summer','summer_autumn','autumn','autumn_winter','winter','winter_spring']
    mses = []
    for season in seasons_str:
        mses.append(compute_seasonal_mse(season))
        print(season+' computed')
    if save:
        np.save('figures/residuals/mses.npy',np.array(mses))
    return mses


def wrapped_plot(mses=np.load('figures/residuals/mses.npy'),s=4,save=0):
    seasons_str = ['spring','spring_summer','summer','summer_autumn','autumn','autumn_winter','winter','winter_spring']
    seasons_ls = list(np.arange(0, 360, 45))
    def wrap(x):
        if not isinstance(x,list):
            x = list(x)
        if x[0] == 0:
            # print(True)
            return list(np.array(x)-360) + x + list(np.array(x)+360)
        else:
            return x+x+x
    season_str_2,mses_2, seasons_ls_2 = wrap(seasons_str),wrap(mses),wrap(seasons_ls)
    rmse = np.sqrt(mses_2)

    plt.figure()
    plt.plot(seasons_ls_2[s:s+8],rmse[s:s+8],color='k')
    plt.scatter(seasons_ls_2[s:s+8],rmse[s:s+8],color='k')
    plt.xticks(seasons_ls_2[s:s+8:2],season_str_2[s:s+8:2])
    plt.ylabel('RMSE (nT)')
    plt.grid(axis = 'x')
    if save:
        plt.savefig(f'figures/residuals/RMSEvsSeason.png', dpi=300)
    else:
        plt.show()

def cdod_vs_ls_grey(mses=np.load('figures/residuals/mses.npy'),save=0,colorbar=0):
    mses=np.load('figures/residuals/mses.npy')
    seasons_ls = list(np.arange(0, 360, 45))
    cdod = []
    for ls in seasons_ls:
        if ls == 0:
            img1 = 'figures/residuals/dust_storms/corrected/ls0_1.png'
            img2 = 'figures/residuals/dust_storms/corrected/ls0_2.png'
            data = 1-np.mean([np.median(gray_scale(img1)),np.median(gray_scale(img2))])
        else:
            img = f'figures/residuals/dust_storms/corrected/ls{ls}.png'
            data = 1-np.median(gray_scale(img))
        cdod.append(data)
     
    p = np.corrcoef(mses, cdod)[0,1]
    print(p)
    plt.figure()
    if colorbar:
        plt.scatter(np.sqrt(mses), cdod,cmap=cmc.vikO,c=seasons_ls,s=100) #[1:]
        add_str = '_lscolor'
    else:
        plt.scatter(np.sqrt(mses), cdod,color='k',s=100)
        add_str = ''
    plt.xlabel('RMSE (nT)')
    plt.ylabel('Mean CDOD')
    if colorbar:
        plt.colorbar(label=r'$L_s$')
    if save:
        plt.savefig(f'figures/residuals/RMSEvsSeason_grayscale{add_str}.png', dpi=300)
    else:
        plt.show()


def cdod_vs_ls_cmap(mses=np.load('figures/residuals/mses.npy'),save=0,colorbar=0):
    mses=np.load('figures/residuals/mses.npy')
    seasons_ls = list(np.arange(0, 360, 45))
    cdod = []
    for ls in seasons_ls:
        if ls == 0:
            img1 = 'figures/residuals/dust_storms/corrected/ls0_1.png'
            img2 = 'figures/residuals/dust_storms/corrected/ls0_2.png'
            data = np.mean([np.median(img2scalar(img1,cmap='YlOrBr')),np.median(img2scalar(img2,cmap='YlOrBr'))])
        else:
            img = f'figures/residuals/dust_storms/corrected/ls{ls}.png'
            data = np.median(img2scalar(img,cmap='YlOrBr'))
        cdod.append(data)
     
    p = np.corrcoef(mses, cdod)[0,1]
    print(p)
    plt.figure()
    if colorbar:
        plt.scatter(np.sqrt(mses), cdod,cmap=cmc.vikO,c=seasons_ls,s=100) #[1:]
        add_str = '_lscolor'
    else:
        plt.scatter(np.sqrt(mses), cdod,color='k',s=100)
        add_str = ''
    plt.xlabel('RMSE (nT)')
    plt.ylabel('Mean CDOD')
    if colorbar:
        plt.colorbar(label=r'$L_s$')
    if save:
        plt.savefig(f'figures/residuals/RMSEvsSeason_cmap{add_str}.png', dpi=300)
    else:
        plt.show()

if __name__=='__main__':
    dummy=0
    # compute_mse_list(1)
    # wrapped_plot()
    cdod_vs_ls_grey(save=0,colorbar=1)
    cdod_vs_ls_cmap(save=0,colorbar=1)

    