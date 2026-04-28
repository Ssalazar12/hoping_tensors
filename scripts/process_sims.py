import numpy as np
import pandas as pd
# from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
# from scipy import sparse
from scipy.optimize import curve_fit

from qutip_tools import *

# In this script we generate a dataset (.csv file) from the raw data

# --------------------------------
# PARAMETERS
# --------------------------------

J = 1
data_route = "../data/sims/L=21/"

# --------------------------------
# Functions
# --------------------------------

def gaussian(x, A, Mu, Sig):
    return A * np.exp(-(x - Mu) ** 2 / (2 * Sig ** 2))

# --------------------------------
# MAIN
# --------------------------------

data_dict = {'L_qpc': [], 'max_time': [], 'tsteps': [], 'bond_index': [],
             'band_width': [], 'k0': [], 'J_prime': [], 't': [], 'Omega': [],
             "vg": [], 'time_at_bond': [], "bond_fit_error": [], "time_f_free": [], "time_f_int": [],
             "xf_avg_free": [], "xf_avg_int": [], "Transmision_tot": [], "Transmission_k0": [],
             "r_density_free": [], "r_density_int": [], "last_density_free": [], "last_density_int": [],
             "last_density_max": [], "time_last_density_max": [], "bond_density_max": [], "min_purity": [],
             "max_VN_entropy": [],
             "entanglement_timeskip": [], "T_mean": [], "ddot0": [], "kick": [],  "theta_f": [], "phi_f": [],
             "delta_phi": [], "dd_density_hit": []}

file_list = os.listdir(data_route)

try:
    file_list.remove('.DS_Store')
except:
    pass

for i in range(0, len(file_list)):

    file_ = file_list[i]

    if (i % 1 == 0):
        print(file_)

    # Load observables
    # remember that when we load the data it is already truncated to the time the QPC hits the end of the chain
    param_dict, times, n_bond, n_left, n_right, n_d1, n_d2, traject, VN_entropy, purity, dd_costheta , dd_sinphi = load_data(data_route, file_)

    # calculate data for the time scales
    tau_L, tau_free, tau_b, vg, x_av, bond_root = get_timescale_data(param_dict, traject, times, n_bond)

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

     # get the occupation of the dot at the time when the qpc reaches it
    tau_to_bond = param_dict["bond_index"]/(2*J*np.sin(param_dict["k0"]))
    tau_to_bond_index = find_nearest_index(times, tau_to_bond)
    dd_density_hit = n_d2[tau_to_bond_index]


    # to estimate the error of how good the time at bond estimation is do a gaussian fit
    # As initial guess of the standard dev put FWHM
    initial_guess = [1, np.mean(n_bond), tau_b]
    params, covariance = curve_fit(gaussian, times, n_bond, p0=initial_guess)
    gauss_fit = gaussian(times, *params)
    # we choose the error measure as the maximum covariance error
    gauss_error = max(np.sqrt(np.diag(covariance)))

    # get transmision probabilities from scattering analytics
    T0, T_tot = get_transmision_proba(param_dict, J)
    # average over the numerical transmision rate (n right)

    # start after the particle has crossed the bond until it hits the wall
    start_time = find_nearest_index(times, tau_to_bond)
    T_mean = np.mean(n_right[start_time:])

    # get the maximum of the density in the last site and the time
    n_max = traject[-1].max()
    time_last_density_max = times[traject[-1].argmax()]

    # calcualte the "kick"
    kick = param_dict["Omega"] * simpson(n_bond, dx=times[1] - times[0])

    # get the change between the initial phi and the final value
    phi0 = np.pi/2 # we always choose this value of phi0
    dd_theta = np.arccos(dd_costheta)
    dd_phi = np.arcsin(dd_sinphi)

    tskip = param_dict["entropy_t_skip"]

    # get the phi at the end of the interaction with the QPC
    times_coarse = np.linspace(0,max(times), len(dd_costheta)) # rtemember the coarse graining in bloch sphere
    # deprecated: measure_end = find_nearest_index(times_coarse, tau_to_bond + tau_b)
    delta_phi = phi0 - dd_phi[-1].real
   
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
    data_dict["bond_fit_error"].append(gauss_error)
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
    data_dict["kick"].append(kick)
    data_dict["theta_f"].append(dd_theta[-1].real)
    data_dict["phi_f"].append(dd_phi[-1].real)
    data_dict["delta_phi"].append(delta_phi)
    data_dict["dd_density_hit"].append(dd_density_hit.real)

data_df = pd.DataFrame.from_dict(data_dict)

print("creating new df")
data_df.to_csv('../data/exp_pro/exploration_data_L={}.csv'.format(param_dict["L_qpc"]))
