import numpy as np
import scipy.optimize as opt
from scipy.ndimage import maximum_filter
from matplotlib.colors import LinearSegmentedColormap

def detect_peaks(data, edge, filter_size=3, order=0.5):
    data_ = data.copy()
    data_[0:int(filter_size*edge), :] = 0
    data_[-int(filter_size*edge):, :] = 0
    data_[:, 0:int(filter_size*edge)] = 0
    data_[:, -int(filter_size*edge):] = 0
    local_max = maximum_filter(data_, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(data_, mask=~(data_ == local_max))
    #Remove small peaks
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max()*order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

#Define gaussian function
def twoD_Gaussian(XY, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = XY[0:2]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

#Calculate cetroid
def centroid(data):
    h,w = np.shape(data)
    x = np.arange(0,w)
    y = np.arange(0,h)
    X,Y = np.meshgrid(x,y)
    cx = np,sum(X*data)/np,sum(data)
    cy = np,sum(Y*data)/np,sum(data)
    return cx, cy

#Fit gaussian
def gauss(data, size_peak, FWHM):
    #Create x and y indices
    x_i = np.linspace(0, size_peak-1, size_peak)
    y_i = np.linspace(0, size_peak-1, size_peak)
    x_i, y_i = np.meshgrid(x_i, y_i)
    data = np.reshape(data, (size_peak*size_peak))
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x_i, y_i), data, p0=(max(data)-min(data), size_peak/2, size_peak/2, FWHM/2.35, FWHM/2.35, 1, min(data)))
    cx = popt[1]
    cy = popt[2]
    data_fitted = twoD_Gaussian((x_i, y_i), *popt)
    return cx, cy

#Make color map
def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v,c in zip(values, colors):
        color_list.append((v/vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)
