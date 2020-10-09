import numpy as np
import sys
import shlex,subprocess
import math
import matplotlib.pyplot as plt
import struct

def parameters_from_input(path_to_solution_folder):
    """
    #----------Description-------------#
    # Functions is designet to fish essential 
    # parameters from input file.2.35456
    #
    #------------Input-----------#
    # path_to_solution_folder - str, path to folder with intresting 
    # result of simulation.
    #
    #-----------Output-----------#
    # delta_t - double, time step between saves of files,
    # delta_x - double, spacial time step,
    # a_rel - double, relatibvistic amplitude,
    # n - double, dimentionless concentration,
    # focal_spot_fwhm - double, FocalSpotWidthFWHM (for energy).
    """
    
    key_phrases = {'delta_x': "DeltaX = ", 'a_rel': "RelativisticField = ", 'n_cr': "Ncr = ", 'n_0': "N = ", 'PerCell': "ElectronsPerCell = ", "time_step": "TimeStep = ", "BOI": "BOIterationPass = ", "focal_spot_fwhm": "FocalSpotWidthFWHM = "} # словарь для типичного входного файла
    
    input_BOI = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['BOI'],path_to_solution_folder) 
    args_BOI = shlex.split(input_BOI)
    BOI = subprocess.check_output(args_BOI, encoding='UTF-8')
    BOI = float(BOI[len(key_phrases['BOI']):-1])
    
    
    input_time_step = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['time_step'],path_to_solution_folder) 
    args_time_step = shlex.split(input_time_step)
    time_step = subprocess.check_output(args_time_step, encoding='UTF-8')
    time_step = float(time_step[len(key_phrases['time_step']):-1])
    delta_t = time_step*math.floor(BOI)
    
    input_delta_x = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['delta_x'],path_to_solution_folder) 
    args_delta_x = shlex.split(input_delta_x)
    delta_x = subprocess.check_output(args_delta_x, encoding='UTF-8')
    delta_x = float(delta_x[len(key_phrases['delta_x']):-1])/100 # /100 need for meters
    
    input_a_rel = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['a_rel'],path_to_solution_folder) 
    args_a_rel = shlex.split(input_a_rel)
    a_rel = subprocess.check_output(args_a_rel, encoding='UTF-8')
    a_rel = float(a_rel[len(key_phrases['a_rel']):-1])
    
    input_n_cr = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['n_cr'],path_to_solution_folder) 
    args_n_cr = shlex.split(input_n_cr)
    n_cr = subprocess.check_output(args_n_cr, encoding='UTF-8')
    n_cr = float(n_cr[len(key_phrases['n_cr']):-1])
    input_n_0 = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['n_0'],path_to_solution_folder) 
    args_n_0 = shlex.split(input_n_0)
    n_0 = subprocess.check_output(args_n_0, encoding='UTF-8')
    n_0 = float(n_0[len(key_phrases['n_0']):-1])
    input_PerCell = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['PerCell'],path_to_solution_folder) 
    args_PerCell = shlex.split(input_PerCell)
    PerCell = subprocess.check_output(args_PerCell, encoding='UTF-8')
    PerCell = float(PerCell[len(key_phrases['PerCell']):-1])
    n = PerCell*n_cr/n_0
    
    input_focal_spot_fwhm = 'grep \"{}\" {}/ParsedInput.txt'.format(key_phrases['focal_spot_fwhm'],path_to_solution_folder) 
    args_focal_spot_fwhm = shlex.split(input_focal_spot_fwhm)
    focal_spot_fwhm = subprocess.check_output(args_focal_spot_fwhm, encoding='UTF-8')
    focal_spot_fwhm = float(focal_spot_fwhm[len(key_phrases['focal_spot_fwhm']):-1])
    
    return delta_t, delta_x, a_rel, n, focal_spot_fwhm

def reparse(path_to_solution_folder, name_of_folder, dimensionless_coefficient, DeltaX):
    """
    #------------Description-----------#
    # Function make numpy.ndarray from binary file with various data.
    # Also all output is dimentionless.
    #
    #----------Input---------------#
    # path_to_solution_folder - str, path to folder with intresting 
    # result of simulation,
    # name_of_folder - str, self explanatory,
    # dimensionless_coefficient - double, coefficient data has to be divided by,
    # DeltaX - double, if you want to have space array pass space step here, else make it zero.
    #
    #----------Output---------------#
    # array_out - numpy.ndarray, dimentionless list of data in each moment of time.
    """
    
    raw_files_list = subprocess.check_output(['ls',], cwd = path_to_solution_folder + '//BasicOutput//data//' + name_of_folder)
    files_list = raw_files_list.split()
    files_list.sort()
    files_list = [i.decode('ascii') for i in files_list]
    array_out = []
    for file in files_list:
        distribution1D = []
        with open(path_to_solution_folder + '//BasicOutput//data//' + name_of_folder +"//"+ file, "rb") as f:
            bite = f.read(4)
            i = 0
            while bite:
                i += 1
                value = struct.unpack('f', bite)[0]
                distribution1D.append(value/dimensionless_coefficient)
                bite = f.read(4)
        array_out.append(distribution1D)
    if DeltaX == 0:
        return np.array(array_out)
    else:
        x_array = np.arange(len(array_out[0]))*DeltaX
        return np.array(array_out), x_array
    
"""a,b,c,d = (parameters_from_input(sys.argv[1]))
bz, x = reparse(sys.argv[1], sys.argv[2], c, b)
ne = reparse(sys.argv[1], sys.argv[3], d, 0)
plt.plot(x, bz[0])
plt.plot(x, ne[0])
plt.show()"""