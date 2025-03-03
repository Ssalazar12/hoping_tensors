import numpy as np
from qutip import  *

import h5py
import json
import os
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
# from scipy import sparse

# HELPER FUNCTIONS FOr python project

# ----------------------------------------------------
# FOR DATA ANALYSIS
# ----------------------------------------------------

def find_nearest_index(array, value):
    # finds the index of the element closest to value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx
def load_data(dir_route, file):
    # loading the simulation data. In addition. it makes sure to implement
    # THE TIME=INFINITY CUTOFF TO AVOID back reflections
    with h5py.File(dir_route + file, 'r') as res_h5:
        J=1
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

        # Here we definide t=infty when the free QPC particle would hit the bond
        # which is calculated like this
        # trajectory using Ehrenfest Therem
        vg = 2 * J * np.sin(Param_dict["k0"])
        tau_free = Param_dict["L_qpc"] / vg
        last_t_index = find_nearest_index(Times, tau_free)

        Times = Times[:last_t_index]
        N_bond = N_bond[:last_t_index]
        N_left = N_left[:last_t_index]
        N_right = N_right[:last_t_index]
        N_d1 = N_d1[:last_t_index]
        N_d2 = N_d2[:last_t_index]
        Trajectories = Trajectories[:,:last_t_index]
        VN_entropy = VN_entropy[:last_t_index]
        Purity = Purity[:last_t_index]

        # Bloch sphere for DD dot_bloch_theta dot_bloch_phi
        # since for entropy and such we coarse grained tim we need this again
        # if there are nans' then eliminated them
        try:
            last_t_index = find_nearest_index(Times[::Param_dict["entropy_t_skip"]], tau_free)
            DD_costheta = res_h5["results/dot_bloch_costheta"][:last_t_index]
            DD_sinphi = res_h5["results/dot_bloch_sinphi"][:last_t_index]
            # when theta = 0 phi becomes undefined because it could take any values
            nan_index = np.argwhere(np.isnan(DD_sinphi))[0][0]
            # replace the nan value with the next numerical val
            DD_sinphi[nan_index] = DD_sinphi[nan_index + 1]
        except IndexError:
            pass

    res_h5.close()

    return Param_dict, Times, N_bond, N_left, N_right, N_d1, N_d2, Trajectories, VN_entropy, Purity, DD_costheta, DD_sinphi


def get_timescale_data(Param_dict, Traject, Times, N_bond):
    # calculates the quantities relevant to the estimation of the hitting time
    # including the time it takes to hit the different parts of the QPC
    J = 1

    QPC_traject = Traject[:, :]  # we only care about qpc trajectories for now
    # vector holding distance to origin of each lattice site
    r_vect = np.arange(0, Param_dict["L_qpc"])
    # position average in time
    x_av = np.asarray([np.dot(QPC_traject[:, i], r_vect) for i in range(0, len(Times))])

    # trajectory using Ehrenfest Therem
    vg = 2 * J * np.sin(Param_dict["k0"])

    # time to get to the bond which is between [bond_index and bond_index+1]
    tau_0b = (Param_dict["bond_index"] - 1) / vg

    # time at the bond defined at width at half maximum of the bond occupation
    # estimate FWHF with an interpolation
    spline = UnivariateSpline(Times, N_bond - np.max(N_bond) / 2, s=0)
    bond_root = spline.roots()  # find the roots
    if (len(bond_root) < 2):
        print("not possible to estimate time at bond for ")
        print(Param_dict)
        tau_b = -Times[-1]
    else:
        # the first two roots yield the width at half maximum
        tau_b = bond_root[1] - bond_root[0]

    #time from bond to the wall
    tau_bL = (Param_dict["L_qpc"] - Param_dict["bond_index"] - 2) / vg
    # total time
    tau_L = tau_0b + tau_b + tau_bL

    # time if there were no potential at bond
    tau_free = Param_dict["L_qpc"] / vg

    return tau_L, tau_free, tau_b, vg, x_av, bond_root

def get_transmision_proba(Param_dict, J):
    # in the limit where we have very localized state the transmision probability is approximately that of the
    # one for K0
    V0 = J - 2 * (Param_dict["Omega"] + Param_dict["J_prime"])
    T0 = 1 / (1 + (V0 / Param_dict["k0"]) ** 2)

    # the momentum distribution
    k_arr = np.linspace(-200, 200, 5000)
    Psi0k_abs = (Param_dict["band_width"] ** 2 / np.pi) ** (1 / 2) * np.exp(
        -(Param_dict["band_width"] ** 2) * (k_arr - Param_dict["k0"]) ** 2)
    # now with the wave packet weights
    T_k = 1 / (1 + (V0 / k_arr) ** 2)
    T_tot = simpson(T_k * Psi0k_abs, dx=k_arr[1] - k_arr[0])

    return T0, T_tot

# ----------------------------------------------------
# FOR CREATING HAMILTONIANS AND OPERATORS IN QUTIP
# ----------------------------------------------------

def get_thight_binding_hamiltonian(op_list, Nsites,jcouple, bc="fixed"):
    # creates the tight binding hamiltonian  from the fermion operators in op_list
    # and the coupling array jcouple with the chosen boundary conditions
    # for Nsites lattices sites

    ident_tensor = tensor([identity(2)]*(Nsites)) 
    H = 0*ident_tensor

    for site_j in range(0,Nsites-1):
        H += -0.5*jcouple[site_j]*(op_list[site_j].dag()*op_list[site_j+1]+op_list[site_j+1].dag()*op_list[site_j])
        
    if bc == "periodic":
        print("periodic")
        # operator that acts on the final 
        # implement periodic boundaries
        H += -0.5*jcouple[Nsites-1]*(op_list[Nsites-1].dag()*op_list[0]+op_list[0].dag()*op_list[Nsites-1])
        
    return H 

def get_initial_state(init_coefs, basis_set):
	# creates the initial Psi0 state by combining the lists init_coefs and basis_set
	# into a normalized qutip ket

	Psi0 = np.sum([init_coefs[j]*basis_list[j] for j in range(0,len(init_coefs))])
	Psi0 = Psi0.unit()

	return Psi0

def get_1p_basis(Nsites):
    # creates the initial wave function from the init_coefs list
    # and the one particle basis vectors
    # create the density matrix from ONE particle basis states
    # list holding all possible 1-particle states
    string_list = []
    basis_list = []

    # MAKE SURE EVERYTHING STAYS SPARSE
    b1 = basis(2,0)
    b1.data = data.to(data.CSR, b1.data)    
    b2 = basis(2,1)
    b2.data = data.to(data.CSR, b2.data)
    
    for site_j in range(0,Nsites):
        # create emty sites
        site_vectors = [b1]*Nsites
        site_string = [0]*Nsites
        
        # create an exitation at site j
        site_vectors[site_j] = b2
        site_string[site_j] = 1
        
        string_list.append(site_string)
        basis_list.append(tensor(site_vectors))

    return string_list, basis_list

def get_2p_basis(Nsites):
    #creates a two particle basis for a lattices of size Nsites
    
    string_list = [] # to track where we put the particles
    basis_list = [] # to save the multiparticle basis states
    for site_i in range(0,Nsites-1):
        for site_j in range(site_i+1,Nsites):
            site_string = [0]*Nsites
            site_vectors = [basis(2, 0)]*Nsites
            # place a fermion in site 0 and another in j
            site_vectors[site_i] = basis(2, 1)
            site_vectors[site_j] = basis(2, 1)
            # track the placement as a string
            site_string[site_i] = 1
            site_string[site_j] = 1

            basis_list.append(tensor(site_vectors))
            string_list.append(site_string)

    return string_list,basis_list


def get_initial_state(init_coefs, basis_set):
    # creates the initial Psi0 state by combining the lists init_coefs and basis_set
    # into a normalized qutip ket
    
    Psi0 = np.sum([init_coefs[j]*basis_set[j] for j in range(0,len(init_coefs))])
    Psi0 = Psi0.unit()
    
    return Psi0

def create_lindblad_op(Nsites, operator_list ,gamma,collapse_type="number"):
    # creates the operators necesary for the non-unitrary dynamics i.e
    # collapse operatos and the operators for the expectation values
    # collapse_type = "number" or "ladder"
    collapse_ops = []
    expect_ops = []

    for site_j in range(0,Nsites):
        density_op = operator_list[site_j].dag()*operator_list[site_j]
        expect_ops.append(density_op)
        
        if collapse_type=="number":
            collapse_ops.append(np.sqrt(gamma)*density_op)
        else: 
            collapse_ops.append(np.sqrt(gamma)*operator_list[site_j])
    
    return collapse_ops, expect_ops

