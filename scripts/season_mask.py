import torch




def get_condition(season = 'summer'):
    if season == 'summer':
        target_ls = 90
    elif season == 'winter':
        target_ls = 270
    elif season == 'spring':
        target_ls = 0
    elif season  == 'autumn':
        target_ls = 180
    elif season == 'summer_autumn':
        target_ls = 135
    elif season == 'autumn_winter':
        target_ls = 225
    elif season == 'winter_spring':
        target_ls = 315
    elif season == 'spring_summer':
        target_ls = 45
    else:
        raise ValueError('season_filter must be "summer", "winter", "spring", "autumn" or None')
    angle_half_band = 30
    ls = torch.load('data/Ls_series.pt')
    lower_bound = target_ls - angle_half_band
    upper_bound = target_ls + angle_half_band
    lower_bound = lower_bound % 360
    upper_bound = upper_bound % 360
    if lower_bound > upper_bound:
        condition1 = (ls >= lower_bound) | (ls <= upper_bound)
    else:
        condition1 = (ls <= upper_bound) & (ls >= lower_bound)

    input_sph = torch.load('data/position_mso_spherical.pt')
    condition2 = (input_sph[:,0] <= 500)
    del input_sph
    condition = condition1 & condition2
    return condition

