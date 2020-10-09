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
    # привязка пространственных частот через известное падающее поле
    rfft_i = abs(np.fft.rfft(field_i, axis = 1))
    argmax = np.argmax(rfft_i, axis = 1)[70]
    
    k_x = np.arange(1,np.shape(rfft_i)[1])/argmax
    lambda_i = 0.8/k_x
    filter_range_i = (np.where(lambda_i>filter_range[1])[0][-1], np.where(lambda_i>filter_range[0])[0][-1])
    
    filtered_fft_r = np.zeros(np.shape(abs(np.fft.rfft(field_r, axis = 1))), dtype = "complex")
    filtered_fft_r[:, filter_range_i[0]:filter_range_i[1]] = (np.fft.rfft(field_r, axis = 1))[:, filter_range_i[0]:filter_range_i[1]]
    
    filtered_r = np.fft.irfft(filtered_fft_r, axis = 1)
    
    return filtered_r
