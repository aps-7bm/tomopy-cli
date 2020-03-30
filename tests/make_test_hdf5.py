import unittest
import h5py
import numpy as np
import matplotlib.pyplot as plt

test_hdf_filename = 'test_dx_180.hdf5'


def make_elliptic_profile(center, radius, density, total_pts):
    '''Make the projection of a circular profile of
        the given center, radius, and density.
        The data will be in a Numpy array total_pts wide.
    '''
    output = np.abs(np.arange(total_pts) - center)
    output[output > radius] = radius
    output = np.sqrt(radius**2 - output**2) * 2 * density
    return output
    

class TestEllipticProfile(unittest.TestCase):
    def test_make_elliptic_profile(self):
        assert np.allclose(0, make_elliptic_profile(500, 50, 0, 1000))
        assert np.allclose(40.0, np.max(make_elliptic_profile(500, 20, 1, 1000)))
        assert np.allclose(20.0, float(np.argmax(make_elliptic_profile(20, 5, 5, 100))))
        #plt.plot(make_elliptic_profile(511.5, 50, 1, 1024), 'r.')
        #plt.show()


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
