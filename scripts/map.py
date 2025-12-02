import pygmt
import pandas as pd
import warnings
import configs.config_map as config
import os
import numpy as np
# import numpy as np
# import geopandas as gpd

warnings.filterwarnings("ignore")

#fontsize
pygmt.config(FONT_ANNOT_PRIMARY='16p')


def one_map(parameter, direction, filetype, save = 1):
    assert isinstance(parameter, str)
    assert isinstance(direction, str)
    model_name = config.map_config['model_name']
    pathinput = 'predictions/'+model_name+'.csv'
    data = pd.read_csv(pathinput, header=0)
    # print(data.describe())

    pathoutput = 'maps/'+model_name
       
    fig = pygmt.Figure()
    region = [-180, 180, -90, 90]
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
        series = [-20,20]
        cmap = 'vik'
        reverse_bool = False
        units = 'nA/m2'

    if direction == 'total':
        try:
            fill = 0.0
            for i in ['r','p','t']: #['x','y','z']: #
                fill += (data[parameter+i].values)**2
            fill = np.sqrt(fill)
        except:
            fill = data[parameter].values
        series = [0,25]
        cmap = 'imola'
        reverse_bool = False
    else:
        fill = data[label].values

    if config.map_config['colorbar_fixed']:
        add_txt = '_fixed_range'
    else:
        series = [fill.min(), fill.max()]
        add_txt = '_free_range'

    pathoutput += add_txt
    os.makedirs(pathoutput, exist_ok=True)

    # plot upper layer
    pygmt.makecpt(cmap=cmap, reverse = reverse_bool, series= series)
    fig.plot(x=data['lon'].values, y=data['lat'].values, style="c0.03c", fill=fill, cmap=True, region=region, projection=projection)
   
    fig.basemap(region=region, projection=projection, frame=frame)
    pygmt.makecpt(cmap=cmap, reverse = reverse_bool, background="o+t", series= series)
    fig.colorbar(frame=["x+l"+label+f" [{units}]"],position="JBC+o0c/1c")
    if save == 1:
        fig.savefig(pathoutput+'/'+label+filetype, dpi = 300)
    else:
        return pathoutput, data, fig

def all_maps():
    if config.map_config['filetype'] == 'both':
        filetypes = ['.png','.pdf']
    else:
        filetypes = [config.map_config['filetype']]
    for filetype in filetypes:
        for parameter in ['B', 'J']:
            if config.map_config['frame'] == 'spherical':
                for direction in ['r', 't', 'p']: #'r', 't', 'p', ,'total'
                    one_map(parameter,direction,filetype)
            elif config.map_config['frame'] == 'cartesian':
                for direction in ['x', 'y', 'z']:
                    one_map(parameter,direction,filetype)

def wind_plot():
    pathoutput, data, fig = one_map('J','total','.png',0)

    angle = np.arctan2(-data.Jt.values, data.Jp.values) * 180 / np.pi  # Convert to degrees
    magnitude = np.sqrt(data.Jp.values**2 + data.Jt.values**2)

    vector_data = np.column_stack([
        data.lon.values,    # x_start
        data.lat.values,    # y_start  
        angle,              # direction_degrees
        np.sqrt(magnitude)/5           # length
    ])

    # keep one point out of 1000 for clarity
    vector_data = vector_data[::1500]

    fig.plot(
    data = vector_data,
    style="v0.1c+eA",
    pen="0.4p",
    # fill="red3",
    )

    fig.savefig(pathoutput+'/J_total_wind.png', dpi = 300)



if __name__=='__main__':

    if config.wind_map:
        wind_plot()

    if config.all_maps:
        all_maps()

    if config.only_B:
        one_map('B','total',config.map_config['filetype'])

    