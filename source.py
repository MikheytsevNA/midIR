import numpy as np
import matplotlib.pyplot as plt

def source(field_i, field_r, filter_range):
    """
    #------------Description-----------#
    # Returns filtered (t,x) matrix of reflected fields.
    #
    #---------Input------------------------#
    # field_i - numpy.ndarray shape - (t,x), icident field,
    # field_r - numpy.ndarray shape - (t,x), reflected field,
    # filter_range - tuple - (lambda_1, lambda_2) - tuple of with intresting spectral range.
    #
    #--------Output-------------------------#
    # filtered_r - numpy.ndarray shape - (t,x), filtered reflected field. 
    
    """
    def filter_gauss(w, shift, sigma):
        return np.exp(-((w-shift)/sigma)**2/2)
    # привязка пространственных частот через известное падающее поле
    rfft_i = abs(np.fft.rfft(field_i, axis = 0))
    argmax = np.unravel_index(np.argmax(rfft_i, axis = None), rfft_i.shape)[0]
    
    k_x = np.arange(1,np.shape(rfft_i)[0])/argmax
    lambda_i = 0.8/k_x
    filter_range_i = (np.where(lambda_i>filter_range[1])[0][-1], np.where(lambda_i>filter_range[0])[0][-1])
    
    filter_shape = filter_gauss(np.arange(len(rfft_i[:,0])), sum(filter_range)/2, (filter_range[1] - filter_range[0])/2)
    filter_shape[filter_range[0]:int(sum(filter_range)/2)] = abs(np.ones(len(field_r)))[filter_range[0]:int(sum(filter_range)/2)]
    filter_shape[0:filter_range[0]] = 0
    filter_2d = np.ones_like(rfft_i)
    for i in range(filter_2d.shape[1]):
        filter_2d[:,i] *= filter_shape
    
    filtered_fft_r = (np.fft.rfft(field_r, axis = 0))*filter_2d
    
    filtered_r = np.fft.irfft(filtered_fft_r, axis = 0)
    
    return filtered_r
