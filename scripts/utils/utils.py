import torch
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import imageio.v3 as iio
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
# import pyshtools as sh

def generate_input_grid(n = 800, save = 0):
    colat_deg = np.linspace(0, 180, n, endpoint=False)
    lon_deg = np.linspace(0, 360, 2*n, endpoint=False)
    colat_deg_mesh, lon_deg_mesh = np.meshgrid(colat_deg, lon_deg, indexing='ij')
    lat_deg_flat = 90-colat_deg_mesh.flatten()
    lon_deg_flat = lon_deg_mesh.flatten()
    df = pd.DataFrame({'lat': lat_deg_flat, 'lon': lon_deg_flat})
    lmax = (n/2) - 1
    print('lmax: ',lmax)
    print('Nb data: ',len(df))
    return df, lmax

def standardize(inp, mean, std):
    '''return (inp-mean)/std'''
    return (inp-mean)/std

def destandardize(inp, mean, std):
    '''return inp * std + mean'''
    return inp * std + mean

def spherical_to_cartesian_torch(r, colat, lon):
    '''input units: r in km or m, colat in radian, lon in radian
    output units: x, y, z like r'''
    x = r*torch.sin(colat) * torch.cos(lon)
    y = r*torch.sin(colat) * torch.sin(lon)
    z = r*torch.cos(colat)
    return torch.stack((x,y,z), dim=1)

def spherical_to_cartesian_np(r, colat, lon):
    '''input units: r in km or m, colat in radian, lon in radian
    output units: x, y, z like r'''
    x = r*np.sin(colat) * np.cos(lon)
    y = r*np.sin(colat) * np.sin(lon)
    z = r*np.cos(colat)
    return x, y, z

def cartesian_to_spherical_torch(x, y, z):
    '''input units: x, y, z in km or m
    output units: r like input, colat in radian, lon in radian
    '''
    r = torch.sqrt(x**2 + y**2 + z**2)
    colat = torch.acos(z/r)
    lon = torch.atan2(y,x)
    return r, colat, lon

def cartesian_to_spherical_np(x, y, z):
    '''input units: x, y, z in km or m
    output units: r like input, colat in radian, lon in radian
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    colat = np.arccos(z/r)
    lon = np.arctan2(y,x)
    return r, colat, lon

def tensorize(t):
    '''Converts numpy array to torch tensor'''
    return torch.tensor(t,dtype = torch.float32)

def compute_area_sphere_sample(lat_deg1, lon_deg1, lat_deg2, lon_deg2, r = 1):
    p1, p2, t1, t2 = np.deg2rad(lon_deg1), np.deg2rad(lon_deg2), (np.pi/2)-np.deg2rad(lat_deg2), (np.pi/2)-np.deg2rad(lat_deg1)
    area = (np.cos(t1) - np.cos(t2))*(p2-p1) * r**2
    return area

def fibonacci_sphere(samples,alt,plot=0):
    '''
    Generates a fibonacci sphere of samples points at altitude alt (in km).
    Output 1: pandas dataframe with columns alt in km, lat in degrees, lon in degrees
    Output 2: torch tensor of shape (samples,3) with cartesian coordinates in km
    '''
    a = 3390 # km
    r = (a + alt) # km
    X,Y,Z = [],[],[]
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        # radius = r

        theta = phi * i  # golden angle increment

        X.append(math.cos(theta) * radius *r)
        Z.append(y*r)
        Y.append(math.sin(theta) * radius*r)

    tensor = torch.stack((tensorize(X), tensorize(Y), tensorize(Z)), dim=1)

    if plot:
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.scatter3D(X,Y,Z, c=Z, cmap='viridis')
        plt.show()

    _,colat, lon = cartesian_to_spherical(tensor[:,0], tensor[:,1], tensor[:,2])
    lat_deg = np.rad2deg(np.pi/2 - colat)
    lon_deg = np.rad2deg(lon)
    df = pd.DataFrame({'alt':alt*np.ones(samples), 'lat':lat_deg, 'lon':lon_deg})
    return df, tensor

def fibonacci_sphere_restricted(samples,alt,lat_deg,lon_deg):
    '''
    Generates a fibonacci sphere of samples points at altitude alt (in km).
    Output 1: pandas dataframe with columns alt in km, lat in degrees, lon in degrees
    Output 2: torch tensor of shape (samples,3) with cartesian coordinates in km
    '''
    ratio = compute_area_sphere_sample(lat_deg[0], lon_deg[0], lat_deg[1], lon_deg[1]) / (4*np.pi)
    n = int(samples/ratio)
    df, tensor = fibonacci_sphere(n, alt)
    lon_0_360 = df['lon'].copy()
    lon_0_360[lon_0_360<0] = 360 + lon_0_360[lon_0_360<0]

    condition = (df['lat'] >= lat_deg[0]) & (df['lat'] <= lat_deg[1]) & (lon_0_360 >= lon_deg[0]) & (lon_0_360 <= lon_deg[1])
    df = df[condition]
    tensor = tensor[condition]
    df.reset_index(drop=True, inplace=True)
    n_prime = len(df)
    return df, tensor, n_prime


def preprocess_input(input_tensor, std_params, coord = 'cartesian'):
    '''input_tensor: torch tensor of shape (n,3)
    std_params: dictionary with keys 'mean_input', 'std_input'
    
    Returns the input tensor with standardized values using preexisting mean and std, and in cartesian coordinates.
    '''
    processed_input = input_tensor.clone()
    if coord == 'spherical':
        processed_input[:,0], processed_input[:,1], processed_input[:,2] = spherical_to_cartesian_torch(processed_input[:,0], processed_input[:,1], processed_input[:,2])
    processed_input = standardize(processed_input, std_params['mean_input'], std_params['std_input'])
    return processed_input

def field_cart_to_spher_torch(Bx,By,Bz,lat_deg=None, lon_deg=None,colat_rad=None,lon_rad=None,device=None):
    '''Converts magnetic field components from cartesian to spherical coordinates. '''
    if (colat_rad is None) & (lon_rad is None):
        try:
            colat_rad = np.pi/2 - np.radians(lat_deg)
            lon_rad = np.radians(lon_deg)
        except:
            print('Please provide (co-)latitude and longitude')
            return
    if device is not None:
        colat = torch.tensor(colat_rad, dtype=torch.float32, device=device)
        lon = torch.tensor(lon_rad, dtype=torch.float32, device=device)
    else:
        colat = torch.tensor(colat_rad, dtype=torch.float32)
        lon = torch.tensor(lon_rad, dtype=torch.float32)
    Br = Bx*torch.sin(colat)*torch.cos(lon) + By*torch.sin(colat)*torch.sin(lon) + Bz*torch.cos(colat)
    Bt = Bx*torch.cos(colat)*torch.cos(lon) + By*torch.cos(colat)*torch.sin(lon) - Bz*torch.sin(colat)
    Bp = -Bx*torch.sin(lon) + By*torch.cos(lon)
    return Br, Bt, Bp

def field_cart_to_spher_np(Bx,By,Bz,lat_deg=None, lon_deg=None,colat_rad=None,lon_rad=None):
    '''Converts magnetic field components from cartesian to spherical coordinates. '''
    if (colat_rad is None) & (lon_rad is None):
        try:
            colat_rad = np.pi/2 - np.radians(lat_deg)
            lon_rad = np.radians(lon_deg)
        except:
            print('Please provide (co-)latitude and longitude')
            return
    colat= colat_rad
    lon = lon_rad
    Br = Bx*np.sin(colat)*np.cos(lon) + By*np.sin(colat)*np.sin(lon) + Bz*np.cos(colat)
    Bt = Bx*np.cos(colat)*np.cos(lon) + By*np.cos(colat)*np.sin(lon) - Bz*np.sin(colat)
    Bp = -Bx*np.sin(lon) + By*np.cos(lon)
    return Br, Bt, Bp


def field_spher_to_cart(input, target):
    '''Converts target from spherical to cartesian coordinates. 
    
    Units: input = r or alt (useless here), colat in rad, lon in rad
    
           target=Br,Bt,Bp in nT
    
    '''
    # r = input[:,0] # km
    colat = input[:,1] # rad
    lon = input[:,2] # rad

    Br = target[:,0] # nT
    Bt = target[:,1] # nT
    Bp = target[:,2] # nT

    Bx = Br*torch.sin(colat)*torch.cos(lon) + Bt*torch.cos(colat)*torch.cos(lon) - Bp*torch.sin(lon) # nT
    By = Br*torch.sin(colat)*torch.sin(lon) + Bt*torch.cos(colat)*torch.sin(lon) + Bp*torch.cos(lon) # nT
    Bz = Br*torch.cos(colat) - Bt*torch.sin(colat) # nT

    return torch.stack((Bx,By,Bz), dim=1)


def slope(k, window):
    y = np.array(k[-window:])
    IQR = np.percentile(y, 75) - np.percentile(y, 25)
    y = np.delete(y, (y > np.mean(y) + 1.5*IQR))
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def pearson_corrcoef(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    num = torch.sum((x - x_mean) * (y - y_mean))
    denom = torch.sqrt(torch.sum((x - x_mean) ** 2)) * torch.sqrt(torch.sum((y - y_mean) ** 2))

    return num / denom

# def td2pyshtools(coeffs_array, lmax, output_name=None, r0 = 3393.5*1000, save = 0):
#     coeffs_array[0] = 0
#     pysh_format = np.zeros((2, lmax+1, lmax+1))
#     index = 0
#     for l in range(lmax+1):
#         for m in range(-l, l+1):
#             if m >= 0:
#                 pysh_format[0, l, m] = coeffs_array[index]
#             else:
#                 pysh_format[1, l, abs(m)] = coeffs_array[index]
#             index += 1
#     coeffs_sh = sh.SHMagCoeffs.from_array(pysh_format, lmax=lmax, r0 = r0)
#     if save:
#         coeffs_sh.to_file('outputs/sh_coefficients/'+output_name+'.sh')
#     return coeffs_sh

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

def mso2lt_lat(ls):
    if not isinstance(ls,int):
        ls = int(ls)

    s_colat = 25.19
    s_lon_all = np.array(range(90, 90-360,-1))% 360
    s_lon = s_lon_all[ls]

    s = np.array([np.sin(np.deg2rad(s_colat))*np.cos(np.deg2rad(s_lon)),
                  np.sin(np.deg2rad(s_colat))*np.sin(np.deg2rad(s_lon)),
                  np.cos(np.deg2rad(s_colat))])
    
    z_mso = np.array([0,0,1])
    rot_angle = s_colat
    rot_axis = np.cross(z_mso, s)/np.linalg.norm(np.cross(z_mso, s))

    K = np.array([[0, -rot_axis[2], rot_axis[1]],
                  [rot_axis[2], 0, -rot_axis[0]],
                  [-rot_axis[1], rot_axis[0], 0]])
    
    R = np.eye(3) + np.sin(np.deg2rad(rot_angle))*K + (1 - np.cos(np.deg2rad(rot_angle)))*(K @ K)

    # sanity check
    assert np.allclose(R.T @ R, np.eye(3)), "Rotation matrix is not orthogonal"
    assert np.linalg.det(R) == 1, "Rotation matrix is not proper"
    assert np.allclose(R @ z_mso, s), "Rotation matrix does not rotate s to z_mso"

    return R.T # passive rotation, rotating the coordinate system


mso2lt_lat(0)