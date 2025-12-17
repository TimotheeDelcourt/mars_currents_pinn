import numpy as np

def mso2lt_lat(ls):
    '''
    Rotation matrix from MSO to (LT, MBF_lat) cartesian frame at a given Ls (degrees). Lon = 0 corresponds to noon.
    Returns a 3x3 numpy array.
    '''
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
