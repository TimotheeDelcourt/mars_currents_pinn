import pygmt
import pandas as pd
import warnings
import configs.config_map as config
import os
# import numpy as np
# import geopandas as gpd

warnings.filterwarnings("ignore")

#fontsize
pygmt.config(FONT_ANNOT_PRIMARY='16p')


def one_map(parameter, direction, filetype):
    assert isinstance(parameter, str)
    assert isinstance(direction, str)
    model_name = config.map_config['model_name']
    pathinput = 'predictions/'+model_name+'.csv'
    data = pd.read_csv(pathinput, header=0)
    # print(data.describe())

    pathoutput = 'maps/'+model_name
    os.makedirs(pathoutput, exist_ok=True)
       
    fig = pygmt.Figure()
    region = [-180, 180, -80, 80]
    projection = "N12c"
    frame = ["xafg90","yafg60"]
    pygmt.config(FONT_ANNOT_PRIMARY='16p')

    label = parameter+direction
    
    if parameter == 'B':
        series = [-30,30]
        cmap = 'roma'
        reverse_bool = True
        units = 'nT'
    elif parameter == 'J':
        series = [-200,200]
        cmap = 'vik'
        reverse_bool = False
        units = 'nA/m2'

    # plot upper layer
    pygmt.makecpt(cmap=cmap, reverse = reverse_bool, series= series)
    fig.plot(x=data['lon'].values, y=data['lat'].values, style="c0.03c", fill=data[label].values, cmap=True, region=region, projection=projection)
   
    fig.basemap(region=region, projection=projection, frame=frame)
    pygmt.makecpt(cmap=cmap, reverse = reverse_bool, series= series, background="o+t")
    fig.colorbar(frame=["x+l"+label+f" [{units}]"],position="JBC+o0c/1c")
    fig.savefig(pathoutput+'/'+label+filetype, dpi = 300)

def all_maps():
    if config.map_config['filetype'] == 'both':
        filetypes = ['.png','.pdf']
    else:
        filetypes = [config.map_config['filetype']]
    for filetype in filetypes:
        for parameter in ['B', 'J']:
            for direction in ['x', 'y', 'z']:
                one_map(parameter,direction,filetype)


if __name__=='__main__':

    all_maps()