import unittest
import h5py
import numpy as np
import matplotlib.pyplot as plt

def make_elliptic_profile(center, radius, density, total_pts):
    '''Make the projection of a circular profile of
        the given center, radius, and density.
        The data will be in a Numpy array total_pts wide.
    '''
    output = np.abs(np.arange(total_pts) - center)
    output[output > radius] = radius
    output = np.sqrt(radius**2 - output**2) * 2 * density
    return output
    

def make_sinogram(angles, num_cols, center_x, center_y, radius, density, cor=None):
    """
    Make a sinogram of a circular region.
    Parameters
    ----------
    angles : Numpy array
        Numpy array of angles for each row of the sinogram, in degrees
    num_cols: int
        Number of colums in the output sinogram
    center_x, center_y: float
        Center location of the circle
    radius: float
        Radius of the circular region
    density: float
        Density of the ciruclar region
    cor: float
        Center of rotation of the sinogram.  Default is center of sinogram
    Returns
    -------
    Numpy array, 2D
        phantom sinogram
    """
    if cor == None:
        cor = num_cols / 2.0 - 0.5
    output = np.zeros((angles.shape[0], num_cols))
    for i,ang in enumerate(np.radians(angles)):
        center_shift = np.cos(ang) * (center_y - cor) - np.sin(ang) * (center_x - cor) 
        output[i,:] = make_elliptic_profile(center_shift + cor, radius, density, num_cols)
    return output
       
angles = np.linspace(0, 180, 181)
sino = make_sinogram(angles, 1000, 500, 500, 100, 1, None)   
plt.imshow(sino)
plt.colorbar()
plt.figure()
sino = make_sinogram(angles, 1000, 400, 500, 100, 2, None)   
plt.imshow(sino)
plt.colorbar()
plt.figure()
plt.plot(sino[0,:],'r.')
plt.plot(sino[90,:],'g.')
plt.show()

class TestEllipticProfile(unittest.TestCase):
    def test_make_elliptic_profile(self):
        assert np.allclose(0, make_elliptic_profile(500, 50, 0, 1000))
        assert np.allclose(40.0, np.max(make_elliptic_profile(500, 20, 1, 1000)))
        assert np.allclose(20.0, float(np.argmax(make_elliptic_profile(20, 5, 5, 100))))


    def test_make_sinogram(self):
        angles = np.linspace(0, 180, 181)
        sino_0 = make_sinogram(angles, 1000, 500, 500, 100, 1, None)   
        assert np.allclose(np.argmax(sino_0[0,:]), 500)
        assert np.allclose(np.argmax(sino_0[-1,:]), 499)
        sino_1 = make_sinogram(angles, 1000, 300, 600, 100, 1, 400)
        assert np.allclose(np.argmax(sino_1[0,:]), 600)
        assert np.allclose(np.argmax(sino_1[90,:]), 500)
        assert np.allclose(np.argmax(sino_1[-1,:]), 200)

def fmake_data_180(fname):
    with h5py.File(fname,'w') as hdf_file:
        hdf_file.create_group('/exchange/')
        theta = np.linspace(0,180,181)
        hdf_file['/exchange'].create_dataset('theta', data = theta)
        


def fmake_meta_data_180(fname):
    with h5py.File(fname,'a') as hdf_file:
        hdf_file.create_group('/measurement/instrument')
        meta_group = hdf_file['/measurement/instrument']
        meta_group.create_dataset('detector/exposure_time',data=[0.1])
        meta_group.create_dataset('detector/brightfield_exposure_time',data=[0.050])
        meta_group.create_dataset('detector/pixel_size_x', data = [4.0])
        meta_group.create_dataset('detection_system/objective/magnification', data = [2.0])


#test_make_elliptic_profile()
#fmake_data_180(test_hdf_filename)
#fmake_meta_data_180(test_hdf_filename)
if __name__ == '__main__':
    unittest.main()
