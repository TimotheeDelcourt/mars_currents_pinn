import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

def img2scalar_nocmap(image,vmin=0,vmax=1):
    '''This function must be used when the colormap of the original figure is not known.
    "Image" should be a path (e.g., "./figure.png")'''
    img = iio.imread(image)
    data = np.dot(img[...,:3]/255, [0.299, 0.587, 0.114])
    data = vmin + data * (vmax - vmin) # denormalize to match vmin and vmax
    return data

def img2scalar_cmap(image,cmap,vmin=0,vmax=1):
    '''This function must be used when the colormap of the original figure is known.
    "Image" should be a path (e.g., "./figure.png"), and cmap simply a str.'''
    try:
        cmap = plt.get_cmap(cmap)
    except:
        cmap = cmc.cmaps(cmap)
    img = iio.imread(image)[...,:3]/255 # RGB in [0,1]
    H, W, _ = img.shape
    cmap_colors = cmap(np.linspace(0,1,256))[:,:3]
    img_flat = img.reshape(-1, 3)
    dists = ((img_flat[:, None, :] - cmap_colors[None, :, :])**2).sum(axis=2) # Compute Euclidean distance to each color in colormap

    # Identify best match
    data = np.argmin(dists, axis=1) # array of integers in [0,255]
    data = data/255.0 # in [0,1]
    data = vmin + data * (vmax - vmin) # denormalize to match vmin and vmax
    data = data.reshape(H,W)
    # print(data)
    return data

