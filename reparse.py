import numpy as np
from scipy.integrate import ode, odeint
from scipy import constants as const
from numpy.fft import fft, rfft, irfft
from scipy.interpolate import griddata
from scipy.special import erf
import struct #для распаковки .bin
import subprocess #для создания списка файлов

class reparse:
    def linear(self, folder_name):
        """
        #----------Description--------
        # Reparse data for experiments with linear polarization.
        #
        #----------Input------------
        # Folder_name - str, absolute path to folder to reparse.
        #
        #----------Output-----------
        #n_0, lambda_i_ad, fft_i_ad, fft_r_ad, lambda_i_sim, fft_r_sim - numpy.ndarray, data
        #
        """
        print(1)
        l = subprocess.check_output(['find', '.', "-name", "LinearPolarization*.bin"], cwd = folder_name)
        folders = l.split()
        folders.sort()
        folders = [i.decode('ascii') for i in folders]
        print(folders[:])

        i=0
        with open(folder_name + "/{}".format(folders[0]), "rb") as f:
                bite = f.read(4)
                while bite:
                    i+=1
                    bite = f.read(4)
        num_t = int(i/6)
        num_n = len(folders)

        n_0 = np.zeros((num_t, num_n))
        lambda_i_ad = np.zeros((num_t, num_n))
        fft_i_ad = np.zeros((num_t, num_n))
        fft_r_ad = np.zeros((num_t, num_n))
        lambda_i_sim = np.zeros((num_t, num_n))
        fft_r_sim = np.zeros((num_t, num_n))

        j = 0
        for folder in folders:
            i = 1
            with open(folder_name + "/{}".format(folder), "rb") as f:
                bite = f.read(4)
                while bite:
                    n_0[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_r_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_i_sim[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_r_sim[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    i += 1
            j += 1
        return     n_0, lambda_i_ad, fft_i_ad, fft_r_ad, lambda_i_sim, fft_r_sim
    def circular_sym(self, folder_name):
        # same as repase.linear
        l = subprocess.check_output(['find', '.', "-name", "symm1*.bin"], cwd = folder_name)
        folders = l.split()
        folders.sort()
        folders = [i.decode('ascii') for i in folders[:-4]]
        print(folders[:])

        i=0
        with open(folder_name + "/{}".format(folders[0]), "rb") as f:
                bite = f.read(4)
                while bite:
                    i+=1
                    bite = f.read(4)
        num_t = int(i/6)
        num_n = len(folders)

        n_0 = np.zeros((num_t, num_n))
        lambda_i_ad = np.zeros((num_t, num_n))
        fft_i_ad = np.zeros((num_t, num_n))
        fft_r_ad = np.zeros((num_t, num_n))
        lambda_i_sim = np.zeros((num_t, num_n))
        fft_r_sim = np.zeros((num_t, num_n))

        j = 0
        for folder in folders:
            i = 1
            with open(folder_name + "/{}".format(folder), "rb") as f:
                bite = f.read(4)
                while bite:
                    n_0[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_r_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_i_sim[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_r_sim[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    i += 1
            j += 1
        return n_0, lambda_i_ad, fft_i_ad, fft_r_ad, lambda_i_sim, fft_r_sim
        return
    def circular_tau(self, folder_name):
        # same as repase.linear
        l = subprocess.check_output(['find', '.', "-name", "tau1*.bin"], cwd = folder_name)
        folders = l.split()
        folders.sort()
        folders = [i.decode('ascii') for i in folders[:-4]]
        print(folders[:])

        i=0
        with open(folder_name + "/{}".format(folders[0]), "rb") as f:
                bite = f.read(4)
                while bite:
                    i+=1
                    bite = f.read(4)
        num_t = int(i/6)
        num_n = len(folders)

        n_0 = np.zeros((num_t, num_n))
        lambda_i_ad = np.zeros((num_t, num_n))
        fft_i_ad = np.zeros((num_t, num_n))
        fft_r_ad = np.zeros((num_t, num_n))
        lambda_i_sim = np.zeros((num_t, num_n))
        fft_r_sim = np.zeros((num_t, num_n))

        j = 0
        for folder in folders:
            i = 1
            with open(folder_name + "/{}".format(folder), "rb") as f:
                bite = f.read(4)
                while bite:
                    n_0[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_r_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_i_sim[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_r_sim[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    i += 1
            j += 1
        return n_0, lambda_i_ad, fft_i_ad, fft_r_ad, lambda_i_sim, fft_r_sim
    def tau_2D(self, folder_name):
        # same as repase.linear
        l = subprocess.check_output(['find', '.', "-name", "2D_*_n0_10.bin"], cwd = folder_name)
        folders = l.split()
        folders.sort()
        folders = [i.decode('ascii') for i in folders]
        print(folders[:])

        i=0
        with open(folder_name + "/{}".format(folders[0]), "rb") as f:
                bite = f.read(4)
                while bite:
                    i+=1
                    bite = f.read(4)
        num_t = int(i/4)
        num_n = len(folders)

        n_0 = np.zeros((num_t, num_n))
        lambda_i_ad = np.zeros((num_t, num_n))
        fft_i_ad = np.zeros((num_t, num_n))
        fft_r_ad = np.zeros((num_t, num_n))
        j = 0
        for folder in folders:
            i = 1
            with open(folder_name + "/{}".format(folder), "rb") as f:
                bite = f.read(4)
                while bite:
                    n_0[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_i_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    fft_r_ad[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    i += 1
            j += 1
        return n_0, lambda_i_ad, fft_i_ad, fft_r_ad

    def analytical_linear(self, folder_name):
        l = subprocess.check_output(['find', '.', "-name", "*_n0_*.bin"], cwd = folder_name)
        folders = l.split()
        folders.sort()
        folders = [i.decode('ascii') for i in folders]
        print(folders[:])
        n_0_list = [x[-8:-4] for x in folders]
        def change_name(old_name, variable):
            print(old_name[:-12])
            print(old_name[-9:])
            return (old_name[:-12] + variable + old_name[-10:])
        data = []
        for folderr in folders:
            folder = change_name(folderr[2:], "t")
            with open(folder_name + "/{}".format(folder), "rb") as file:
                t = np.frombuffer(file.read(), dtype = np.float32)

            folder = change_name(folderr[2:], "x")
            with open(folder_name + "/{}".format(folder), "rb") as file:
                x = np.frombuffer(file.read(), dtype = np.float32)

            folder = change_name(folderr[2:], "dxdt")
            with open(folder_name + "/{}".format(folder), "rb") as file:
                dxdt = np.frombuffer(file.read(), dtype = np.float32)
                
            folder = change_name(folderr[2:], "az_r")
            with open(folder_name + "/{}".format(folder), "rb") as file:
                az_r = np.frombuffer(file.read(), dtype = np.float32)
                
            folder = change_name(folderr[2:], "az_i")
            with open(folder_name + "/{}".format(folder), "rb") as file:
                az_i = np.frombuffer(file.read(), dtype = np.float32)
                
            data.append([t,x,dxdt, az_i, az_r])
        data = np.array(data, dtype=object)
        return n_0_list, data[:,0], data[:,1], data[:,2], data[:,3], data[:,4] # t, x, dxdt, no az_r, az_i
    
    def analytical(self, folder_name):
        l = subprocess.check_output(['find', '.', "-name", "*_n0_*.bin"], cwd = folder_name)
        folders = l.split()
        folders.sort()
        folders = [i.decode('ascii') for i in folders]
        print(folders[:])

        i=0
        with open(folder_name + "/{}".format(folders[0]), "rb") as f:
                bite = f.read(4)
                while bite:
                    i+=1
                    bite = f.read(4)
        num_t = int(i/5)
        num_n = len(folders)

        n_0 = np.zeros((num_t, num_n))
        t = np.zeros((num_t, num_n))
        x = np.zeros((num_t, num_n))
        dxdt = np.zeros((num_t, num_n))
        az_r = np.zeros((num_t, num_n))
        az_i = np.zeros((num_t, num_n))
        j = 0
        for folder in folders:
            i = 1
            with open(folder_name + "/{}".format(folder), "rb") as f:
                bite = f.read(4)
                while bite:
                    n_0[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    t[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    x[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    dxdt[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    az_r[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    az_i[num_t-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    i += 1
            j += 1
        return n_0, t, x, dxdt, az_r, az_i, num_n
    def energy(self, path, to_find):    
        folder_name = path
        l = subprocess.check_output(['find', '.', "-name", to_find], cwd = folder_name)
        folders = l.split()
        #folders.sort()
        folders = [i.decode('ascii') for i in folders]
        print(folders[:])

        i=0
        with open(folder_name + "/{}".format(folders[0]), "rb") as f:
                bite = f.read(4)
                while bite:
                    i+=1
                    bite = f.read(4)
        num_l = int(i/3)
        num_n = len(folders)

        n_0 = np.zeros((num_l, num_n))
        lambda_s = np.zeros((num_l, num_n))
        energy = np.zeros((num_l, num_n))
        j = 0
        for folder in folders:
            i = 1
            with open(folder_name + "/{}".format(folder), "rb") as f:
                bite = f.read(4)
                while bite:
                    n_0[num_l-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    lambda_s[num_l-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    energy[num_l-i][j] = struct.unpack('f', bite)[0]
                    bite = f.read(4)
                    i += 1
            j += 1
        return n_0, lambda_s, energy, num_n
    def fields_long(self, path,files):
        """
        Распаковывает 1D поля.

        Аргументы
        ----------
        path: str, путь до папки с всеми полями,
        files: 1D array of str, список файлов с полями,

        Возвращаемые значения
        -----------
        ez_field_2:  1D array, 1D поле

        """
        ez_field_2 = []
        ey_field_2 = []
        for file in files:
            ez_mom_field = []
            with open(path + 'bz2dxy/' + file, "rb") as f:
                bite = f.read(4)
                i = 0
                while bite:
                    i += 1
                    value = struct.unpack('f', bite)[0]
                    ez_mom_field.append(value/1.33872e+08)# делить на релятивистскую амплитуду
                    bite = f.read(4)
            ez_field_2.append(ez_mom_field)

            ez_mom_field = []
            with open(path + 'by2dxy/' + file, "rb") as f:
                bite = f.read(4)
                i = 0
                while bite:
                    i += 1
                    value = struct.unpack('f', bite)[0]
                    ez_mom_field.append(value/1.33872e+08)# делить на релятивистскую амплитуду
                    bite = f.read(4)
            ey_field_2.append(ez_mom_field)

        return ez_field_2, ey_field_2
    def angle_2D_energy(self, theta, energy_r, data_file_name):
        """
        #----------Description-----------#
        # Reparce energy from file
        #
        #---------Input------------------#
        # data_file_name - str, self explanatory.
        # theta, energy_i, energy_r - blank lists.
        #
        #----------Output------------------#
        # max_Ei - float, maximum energy of incident pulse,
        # theta - numpy.ndarray shape (1, 20*t), angles,
        # energy_r - numpy.ndarray shape (1, 20*t), energies of incident and reflected pulse devided by max_Ei.

        """
        with open(data_file_name, "r") as file:
            theta.append(0)
            energy_r.append(0)
            check = True
            for line in file:
                theta.append(0)
                energy_r.append(0)
                if check == True:
                    print(line)
                    max_Ei, theta[-1], energy_r[-1] = (line.rstrip()).split("    ")
                    check = False
                else:
                    theta[-1], energy_r[-1] = (line.rstrip()).split("    ")
                theta[-1] = float(theta[-1])
                energy_r[-1] = float(energy_r[-1])
        theta = np.array((theta))
        energy_r = np.array((energy_r))
        return max_Ei, theta, energy_r