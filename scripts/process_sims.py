import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
from scipy import sparse

import h5py
import json
import os

from qutip import  *

# In this script we generate a dataset (.csv file) from the raw data

# --------------------------------
# PARAMETERS
# --------------------------------

J=1
data_route = "../data/sims/L=14/"

# --------------------------------
# Functions 
# --------------------------------

def load_data(dir_route, file):
    # loading the simulation data
    with h5py.File(dir_route+file, 'r') as res_h5:
        Param_dict = json.loads(res_h5['metadata/parameters'][()])

        # load qpc data
        N_bond = res_h5["results/QPC_bond_density"][:]
        N_left = res_h5["results/QPC_left_density"][:]
        N_right = res_h5["results/QPC_right_density"][:]

        # load dot data
        N_d1 = res_h5["results/d1_density"][:]
        N_d2 = res_h5["results/d2_density"][:]

        # time range
        Times = res_h5["results/time"][:]
        
        # trajectories
        Trajectories = res_h5["results/trajectories"][:]
        
        # entanglement
        VN_entropy = res_h5["results/dot_VN_entropy"][:]
        Purity = res_h5["results/dot_purity"][:]  

    res_h5.close()
    
    return Param_dict,Times, N_bond, N_left, N_right, N_d1, N_d2,Trajectories, VN_entropy, Purity
    

def get_timescale_data(Param_dict, Traject, Times, N_bond):
    # calculates the quantities relevant to the estimation of the hitting time
    # including the time it takes to hit the different parts of the QPC
    J = 1
    
    QPC_traject = Traject[:,:] # we only care about qpc trajectories for now
    # vector holding distance to origin of each lattice site
    r_vect = np.arange(0,Param_dict["L_qpc"])
    # position average in time
    x_av = np.asarray([np.dot(QPC_traject[:,i],r_vect) for i in range(0,len(Times)) ])

    # trajectory using Ehrenfest Therem
    vg = 2*J*np.sin(Param_dict["k0"])
    
    # time to get to the bond which is between [bond_index and bond_index+1]
    tau_0b = (Param_dict["bond_index"]-1)/vg

    # time at the bond defined at width at half maximum of the bond occupation
    # estimate FWHF with an interpolation
    spline = UnivariateSpline(Times, N_bond-np.max(N_bond)/2, s=0)
    bond_root = spline.roots() # find the roots
    if (len(bond_root)<2):
        print("not possible to estimate time at bond for ")
        print(Param_dict)
        tau_b = -Times[-1]
    else:
        # the first two roots yield the width at half maximum
        tau_b= bond_root[1] - bond_root[0]

    #time from bond to the wall
    tau_bL = (Param_dict["L_qpc"]-Param_dict["bond_index"]-2)/vg
    # total time
    tau_L = tau_0b + tau_b + tau_bL

    # time if there were no potential at bond
    tau_free = Param_dict["L_qpc"]/vg
    
    return tau_L, tau_free, tau_b, vg, x_av, bond_root 


def get_transmision_proba(Param_dict, J):
    # in the limit where we have very localized state the transmision probability is approximately that of the
    # one for K0
    V0 = J - 2*(Param_dict["Omega"] + Param_dict["J_prime"])
    T0 = 1/(1+(V0/Param_dict["k0"])**2)

    # the momentum distribution
    k_arr = np.linspace(-200,200, 5000)
    Psi0k_abs = (Param_dict["band_width"]**2/np.pi)**(1/2)*np.exp(-(Param_dict["band_width"]**2)*(k_arr-Param_dict["k0"])**2)
    # now with the wave packet weights
    T_k = 1/(1+(V0/k_arr)**2)
    T_tot = simpson(T_k*Psi0k_abs, dx=k_arr[1] - k_arr[0])
    
    return T0, T_tot
    
def find_nearest_index(array, value):
    # finds the index of the element closest to value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return idx


def get_partial_trace(Psi,NN):
    # calcualtes the partial trace from a STATEVECTOR Psi
    # NOT from the density matrix
    # Psi: Quobject from Qutip representing the wavefunction
    # NN: integer with the total size of the lattice

    n = 2**(NN-2) # QPC sites
    m = 2**2 # Double dot sites
    # get density matrix as a sparse array
    ps = sparse.csr_matrix(Psi.full())
    A = sparse.kron(ps, np.conj(ps.T))
    # convert to normal array for partial trace operation
    Adense = A.toarray()
    # trace out QPC sites and return reduced rho for DD as Quobject
    return Qobj(np.trace(Adense.reshape(n,m,n,m), axis1=0, axis2=2))


# --------------------------------
# MAIN 
# --------------------------------

data_dict = {'L_qpc': [],'max_time': [],'tsteps': [],'bond_index': [],
             'band_width': [],'k0': [],'J_prime': [],'t': [],'Omega': [],
            "vg":[],'time_at_bond':[], "time_f_free":[], "time_f_int": [], 
            "xf_avg_free":[], "xf_avg_int":[], "Transmision_tot":[],"Transmission_k0":[],
            "r_density_free":[], "r_density_int":[], "last_density_free":[],"last_density_int":[],
            "last_density_max":[], "time_last_density_max":[],"bon":[] ,"min_purity":[], "max_VN_entropy":[],
            "entanglement_timeskip":[], "T_mean":[], "ddot0": []}

file_list = os.listdir(data_route)

try:
    file_list.remove('.DS_Store')
except:
    pass

for i in range(0,len(file_list)):

    file_ = file_list[i]
    
    if(i%1==0):
        print(file_)
    
    # Load observables
    param_dict, times, n_bond, n_left, n_right, n_d1, n_d2, traject,VN_entropy, purity = load_data(data_route,file_)
    
    # bw =3.0 causes problems sinces its super delocalized in space so skip
    if(param_dict["band_width"]==3.0):
        continue

    # calculate data for the time scales
    tau_L, tau_free, tau_b, vg, x_av, bond_root  = get_timescale_data(param_dict, traject, times, n_bond)   
    
    # find time index nearest to the hitting time with interaction
    time_f_i = find_nearest_index(times, tau_L)
    xf_avg_int = x_av[time_f_i]
    r_density_int = n_right[time_f_i]
    last_density_int = traject[-1][time_f_i]

    # find time index nearest to the hitting time in the free case
    time_f_i = find_nearest_index(times, tau_free)
    xf_avg_free = x_av[time_f_i]
    r_density_free = n_right[time_f_i]
    last_density_free = traject[-1][time_f_i]

    # get transmision probabilities from scattering analytics
    T0, T_tot = get_transmision_proba(param_dict, J)
    # average over the numerical transmision rate (n right) 
    # for the final times (time in tending to infinity)
    # find the index corresponding to t=6
    start_time = find_nearest_index(times, 8)
    T_mean = np.mean(n_right[start_time:])

    # get the maximum of the density in the last site and the time
    n_max = traject[-1].max()
    time_last_density_max = times[traject[-1].argmax()]

    data_dict["vg"].append(vg)
    data_dict["time_at_bond"].append(tau_b)

    data_dict["time_f_int"].append(tau_L)
    data_dict["xf_avg_int"].append(xf_avg_int)
    data_dict["r_density_int"].append(r_density_int)
    data_dict["last_density_int"].append(last_density_int)

    data_dict["time_f_free"].append(tau_free)
    data_dict["xf_avg_free"].append(xf_avg_free)
    data_dict["r_density_free"].append(r_density_free)
    data_dict["last_density_free"].append(last_density_free)

    data_dict["Transmision_tot"].append(T_tot)
    data_dict["Transmission_k0"].append(T0)
    data_dict["T_mean"].append(T_mean)

    data_dict["last_density_max"].append(n_max)
    data_dict["time_last_density_max"].append(time_last_density_max)
    data_dict["bond_density_max"].append(np.max(n_bond))
    
    data_dict["min_purity"].append(min(purity))
    data_dict["max_VN_entropy"].append(max(VN_entropy))
    data_dict["entanglement_timeskip"].append(param_dict["entropy_t_skip"])

    data_dict["L_qpc"].append(param_dict["L_qpc"])
    data_dict["bond_index"].append(param_dict["bond_index"])
    data_dict["max_time"].append(param_dict["max_time"])
    data_dict["tsteps"].append(param_dict["tsteps"])
    data_dict["band_width"].append(param_dict["band_width"])
    data_dict["k0"].append(param_dict["k0"])
    data_dict["J_prime"].append(param_dict["J_prime"])
    data_dict["t"].append(param_dict["t"])
    data_dict["Omega"].append(param_dict["Omega"])
    data_dict["ddot0"].append(param_dict["ddot0"])

data_df = pd.DataFrame.from_dict(data_dict)


print("creating new df")
data_df.to_csv('../data/exp_pro/exploration_data_L={}.csv'.format(param_dict["L_qpc"]))
