import numpy as np
from hcipy import *
import matplotlib.pyplot as plt

# --------------------------------------------- #
def Get_PSF(field, wl, f, B):
    '''
    Returns the focal image and the focal grid of a field
    
    :param field: 
    :param wl: wavelength in um
    :param f: focal length in meters
    :param B: distance between the mirrors
    
    :return focal_image:
    :return focal_grid:
    '''
    pupil_grid = make_pupil_grid(1024,diameter=B+4)
    res = wl * f / B 
    wavefront = Wavefront(electric_field=field, wavelength=wl)
    focal_grid = make_focal_grid(q=8, num_airy=16, spatial_resolution=res)
    prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=f)
    focal_image = prop.forward(wavefront)
    
    return focal_image, focal_grid

# --------------------------------------------- #
def Get_OTF(focal_image):
    '''
    Returns the OTF of a given focal image
    It returns the Real OTF
    
    :param focal_image:
    
    :return real_otf:    
    '''
    otf = np.fft.fft2(focal_image.intensity.shaped)
    otf = np.fft.fftshift(otf)
    real_otf = abs((otf))
    return real_otf

# --------------------------------------------- #
def cart2pol(x, y):
    '''
    Converts Cartesian coordinates (x,y) into Polar coordinates (rho,phi)
    :param x: x coordinates. Can be a list, an array, or a number
    :param y: y coordinates. Can be a list, an array, or a number
    
    :return rho,phi: The set of Polar coordinates
    '''
    return(np.sqrt(x**2 + y**2), np.arctan2(y, x))

def pol2cart(rho, phi):
    '''
    Converts Polar coordinates (rho,phi) into Cartesian coordinates (x,y) 
    :param rho: rho coordinates. Can be a list, an array, or a number
    :param phi: phi coordinates. Can be a list, an array, or a number
    
    :return x,y: The set of Cartesian coordinates
    '''    
    return (rho * np.cos(phi), rho * np.sin(phi))

# --------------------------------------------- #
def Make_Single_Aperture(D):
    '''
    Creates a single aperture given the telescope diameter 
    
    :param D: The aperture of the telescope, in meter. float
    
    :return telescope_pupil:
    '''    
    pupil_grid = make_pupil_grid(256,D+2)

    telescope_pupil_generator = make_circular_aperture(D)
    telescope_pupil = telescope_pupil_generator(pupil_grid)

    return telescope_pupil, pupil_grid

# --------------------------------------------- #
def piston(A,phi=None):
    if phi is None:
        phi = np.random.uniform(low=0., high=2*np.pi)
    return A*np.exp(1j*phi)

# --------------------------------------------- #
def Interferometer(x, y, D, rot_angle=0., seeing=None, exposure_time=None, shape="circular"):
    '''
    Makes a N Telescope Interferometer given the position, size of the telescopes.
    Can rotate the interferometer using rot_angle in radians.
    Can add seeing or not using seeing.
    
    :param x: x coordinates. Can be a list, an array, or a number
    :param y: y coordinates. Can be a list, an array, or a number
    :param D: The aperture of the telescopes, in meter. float
    :param rot_angle: Angle by which you want to rotate your interferometer. In radians. float
    :param seeing: Add seeing if True, don't add seeing if None. Boolean
    :param exposure_time: Exposure time in ms. Must be an int. If None exposure time is snapshot => 1ms. 
    
    :return field:
    :return pupil_grid: 
    '''
    field=0.
    pupil_grid = make_pupil_grid(256,[2*(max(x)+2*D),2*(max(y)+2*D)])

    for i in range(len(x)):
        if shape == "circular":
            telescope_pupil_generator = make_rotated_aperture(make_circular_aperture(D,(x[i],y[i])), rot_angle)
        elif shape == "rectangular":
            telescope_pupil_generator = make_rotated_aperture(make_rectangular_aperture(D,(x[i],y[i])), rot_angle)
        else:
            print("Error the shape given should be \"circular\" or \"rectangular\".")
        
        telescope_pupil = telescope_pupil_generator(pupil_grid)

        if seeing:
            piston_telescope_pupil = 0.
            if exposure_time:
                for t in range (0,exposure_time,1): 
                    piston_telescope_pupil += piston(telescope_pupil)
            else:
                piston_telescope_pupil = piston(telescope_pupil)
        
            field += piston_telescope_pupil
        
        else:
            field += telescope_pupil

    return field, pupil_grid

def exp(phi):
    return np.exp(1j*phi)


def new_Interferometer(x, y, D, rot_angle=0., seeing=None, exposure_time=None, wind_speed=None, shape="circular"):
    '''
    Makes a N Telescope Interferometer given the position, size of the telescopes.
    Can rotate the interferometer using rot_angle in radians.
    Can add seeing or not using seeing.
    
    :param x: x coordinates. Can be a list, an array, or a number
    :param y: y coordinates. Can be a list, an array, or a number
    :param D: The aperture of the telescopes, in meter. float
    :param rot_angle: Angle by which you want to rotate your interferometer. In radians. float
    :param seeing: Add seeing if True, don't add seeing if None. Boolean
    :param exposure_time: Exposure time in ms. Must be an int. If None exposure time is snapshot => 1ms. 
    
    :return field:
    :return pupil_grid: 
    '''
    # convert wind speed [m/s] in [m/ms]
    wind_speed *= 1e-3  
    
    field=0.
    pupil_grid = make_pupil_grid(256,[2*(max(x)+2*D),2*(max(y)+2*D)])
    
    yaxis = exposure_time*7
#     array = np.random.rand(7,yaxis)
    array = np.random.uniform(low=0., high=2*np.pi, size=(7, yaxis))
    exp_array = np.vectorize(exp)
    turbuluence = exp_array(array)
        
    for t in range (0,exposure_time,1): 

        piston_telescope_pupil = 0.
        
        for i in range(len(x)):
            if shape == "circular":
                telescope_pupil_generator = make_rotated_aperture(make_circular_aperture(D,(x[i],y[i])), rot_angle)
            elif shape == "rectangular":
                telescope_pupil_generator = make_rotated_aperture(make_rectangular_aperture(D,(x[i],y[i])), rot_angle)
            else:
                print("Error the shape given should be \"circular\" or \"rectangular\".")
                
            telescope_pupil = telescope_pupil_generator(pupil_grid)

            piston_telescope_pupil +=  turbuluence[int( x[i]/D ), int( y[i]/D + (wind_speed*t) )] * telescope_pupil
            
        field += piston_telescope_pupil
         

    return field, pupil_grid