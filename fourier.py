import numpy as np
import matplotlib.pyplot as plt

def incident_and_reflected_fields(field, slice_x, point_t):
    """
    #------------Description-----------#
    # Deivde general field to reflected and incident.
    #
    #---------Input------------------------#
    # field - numpy.ndarray shape - (t,x), field to split,
    # slice_x - int, slice x,
    # point_t - int, point t.
    #
    #--------Output-------------------------#
    # field_i - numpy.ndarray shape (1,t), incident field,
    # field_r - numpy.ndarray shape (1,t), reflected field.
    
    """
    
    field_r = field[:, slice_x].copy()
    field_i = field[:, slice_x].copy()
    
    field_r[:point_t] = 0
    field_i[point_t:] = 0    
    
    return field_i, field_r


def fourier_of_field(field_base, field_addon, lambda_base, lambda_check):
    """
    #------------Description-----------#
    # Do discrete fourier of field
    #---------Input------------------------#
    # field_base - numpy.ndarray shape - (1,t), field with known maximum abs of spectrum,
    # field_addon - numpy.ndarray shape - (1,t), field to procces,
    # lambda_base - double, wavelength corresponding to main frequency,
    # lambda_check - bool, do you need proper lambda fourier? 
    #
    #--------Output-------------------------#
    # fourier_w_base, fourier_w_addon - numpy.ndarray shape - (1, 20*t), normalized to base frequency fourier of input fields,
    # 20 - coefficient for better low frequencies analysis (random, but had to be more than 1)
    # omega - numpy,ndarray shape - (1, 20*t), frequency axis,
    # fourier_l_base, fourier_l_addon - numpy.ndarray shape - (1, 20*t), normalized to base wavelength fourier of input fields,
    # flambda - numpy,ndarray shape - (1, 20*t-1), wavelength axis.
    
    """
    
    fourier_w_base = np.fft.rfft(field_base, n = len(field_base)*20)
    fourier_w_addon = np.fft.rfft(field_addon, n = len(field_addon)*20)
    omega = np.arange(len(fourier_w_base))/np.argmax(abs(fourier_w_base))
    
    flambda = np.array([lambda_base/i for i in omega[1:]])
    if lambda_check == False:
        return fourier_w_base[1:]/max(abs(fourier_w_base)), fourier_w_addon[1:]/max(abs(fourier_w_base)), max(abs(fourier_w_base)), omega[1:], flambda
    else:
        fourier_l_base = fourier_w_base[1:]/flambda**2
        fourier_l_addon = fourier_w_addon[1:]/flambda**2

        return fourier_w_base[1:]/max(abs(fourier_w_base)), fourier_w_addon[1:]/max(abs(fourier_w_base)), max(abs(fourier_w_base)), omega[1:], flambda, fourier_l_base/max(abs(fourier_l_base)), fourier_l_addon/max(abs(fourier_l_base))


def spec_filter(fourier_of_field, flambda, lambda_gate, form_filter):
    """
    #----------Description-----------#
    # Filter input spectrum
    #
    #---------Input------------------#
    # fourier_of_field - numpy.ndarray shape - (1, 20*t), self explanatory,
    # flambda - numpy.ndarray shape - (1, 20*t-1), wavelength axis,
    # lambda_gate - tuple, min and max lambdas,
    # form_filter - str, form of filter i.e. "tophat", "gauss".
    #
    #----------Output------------------#
    # filtered_fourier - numpy.ndarray shape - (1, 20*t-1), filtered field in given spectral bandwidth.
    
    """
    def filter_gauss(w, shift, sigma):
        return np.exp(-((w-shift)/sigma)**2/2)
    if form_filter == "tophat":
        lfilter = np.where((flambda < lambda_gate[0]) | (flambda > lambda_gate[1]))
        filtered_fourier = fourier_of_field.copy()
        filtered_fourier[lfilter] = 0
        return filtered_fourier
    if form_filter == "smooth":
        filter_shape = filter_gauss(np.arange(len(fourier_of_field)), sum(lambda_gate)/2, (lambda_gate[1] - lambda_gate[0])/2)
        filter_shape[lambda_gate[0]:int(sum(lambda_gate)/2)] = abs(np.ones(len(field_r)))[lambda_gate[0]:int(sum(lambda_gate)/2)]
        filter_shape[0:lambda_gate[0]] = 0
        return fourier_of_field*filter_shape
    
    
def reverse_fourier(fourier_of_field, flambda, lambda_gate, form_filter):
    """
    #----------Description-----------#
    # F^(-1)
    #
    #---------Input------------------#
    # fourier_of_field - numpy.ndarray shape - (1, 20*t), self explanatory(had to be NOT normilized(raw rfft)),
    # flambda - numpy.ndarray shape - (1, 20*t-1), wavelength axis,
    # lambda_gate - tuple, min and max lambdas,
    # form_filter - str, form of filter i.e. "tophat", "gauss".
    #
    #----------Output------------------#
    # field - numpy.ndarray shape - (1, t), filtered field in given spectral bandwidth.
    
    """
    if form_filter == "tophat":
        lfilter = np.where((flambda < lambda_gate[0]) | (flambda > lambda_gate[1]))
    filtered_fourier = fourier_of_field.copy()
    filtered_fourier[lfilter] = 0
    """
    test
    plt.plot(flambda, abs(fourier_of_field))
    plt.plot(flambda, abs(filtered_fourier))
    plt.axis([0.1,5,0,800])
    """
    field = np.fft.irfft(filtered_fourier)
    return field