
"""
Solves the QPC qubit system via our own exact diagonalization code an saves the relevant quantitites
as an hdf5 file. Also has the capability of doing parameter sweeps 

"""

import sys
import gc
import os
from ast import literal_eval

sys.path.append('../scripts') 

import numpy as np
import h5py
import json

import qutip
from qutip_tools import *

from tqdm import tqdm


# ---------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------

ll = 50
L_qpc_list = [ll]
Omega_list = [0.0, 0.1, 0.3, 0.5]
t_list = [0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4 ,0.5 ,0.6, 0.8 ,1.0, 2.0]
J_prime_list  = [1]
bond_index_list  = [int(ll/2)] # 7
K0_list  = [np.pi/2, 0.95*np.pi/2, 0.9*np.pi/2, 0.8*np.pi/2, 0.7*np.pi/2, 0.6*np.pi/2 , 0.5*np.pi/2,  
			0.4*np.pi/2,  0.3*np.pi/2]
centered_at_list  = [10] # initial position of wavepacket
Delta_list  = [6.0] # spread of wavepacket
maxt_time_list  = [60] # 18 fixed is set by the qpc velocity
N_timepoints_list  = [400]
ddot_list = ["old"] # can be "free", "momentum" OR "fixed" "old" (orbit) which is set by k0 based on af
phi_list = [0] #np.pi/2 # initial phase of the qubit
# if its "free" af, bf will be the initial conditions
af_list = [np.sqrt(0.2)] # np.sqrt(0.8) probability of qubit 0 state

# this is just to get the number of params for the combinations later
Nparams = 13
# data_route = "/home/user/santiago.salazar-jaramillo/hoping_tensors/data/exact_diag_new/L={}/".format(ll)
data_route = "../data/exact_diag_new/L={}/".format(ll)

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------


def create_hamiltonians(L, T, Bond):
    # creates the decoupled and interacting hamiltonians 
    
    # L_qpc = qpc lattice sites
    # T = qubit hopping
    # BOnd = index for bond locatin
    H_matrix = np.zeros((2*L,2*L))
    # fill in the dd hopping 
    d_indices= kth_diag_indices(H_matrix,1)
    H_matrix[d_indices] = -T
    
    # fill in the QPC hopping
    d_indices= kth_diag_indices(H_matrix,2)
    H_matrix[d_indices] = -J[0]
    
    # when qpc and qubit hop a the same time there is no contribution
    d_indices= kth_diag_indices(H_matrix,1)
    odd_inds = (d_indices[0][1::2], d_indices[1][1::2])
    H_matrix[odd_inds] = 0
    
    # save the free hamiltonian for later use
    Hdeco = H_matrix.copy()
    
    # Fill in the interaction at the bond
    H_matrix[2*Bond,2*(Bond+1)] = H_matrix[2*Bond,2*(Bond+1)]+ Omega
    
    # Now the elemets below the diagonal
    for i in range(0,2*L):
        for j in range(i + 1, 2*L):
            H_matrix[j, i] = H_matrix[i, j]
            Hdeco[j, i] = Hdeco[i, j]
            
    return H_matrix, Hdeco


def get_bands(Eigen_energies, Eigen_vectors, Minus_indices, Plus_indies):
    # separates eigenvectors and eigenvalues into plus/minus enegy bands
    # and then sorts by enegy magnitude
    
    # minus band
    Energies_m = Eigen_energies[Minus_indices]
    States_m = Eigen_vectors[:,Minus_indices]
    # sort by magnitude
    Energies_m, States_m = mag_sort(Energies_m, States_m)
    
    # plus band
    Energies_p = Eigen_energies[Plus_indies]
    States_p = Eigen_vectors[:,Plus_indies]
    # sort by magnitude
    Energies_p, States_p = mag_sort(Energies_p, States_p)
    
    return Energies_m, States_m, Energies_p, States_p

    
def get_DD_init_for_momentum(k_prime, alphaf, betaf, ϕ):
    # set initial condtions for a given k_prime such that n_f is always the same
    # k_prime: float. The momentum of the qpc particle
    # nf: The occupations in the qubit 0 state
    
    Tau_bond = -(bond_index)/(2*J[0]*np.sin(k_prime))
    α0 = alphaf*np.cos(Tau_bond*t) + 1j*np.exp(-1j*ϕ)*betaf*np.sin(Tau_bond*t)
    β0 = 1j*np.exp(1j*ϕ)*alphaf*np.sin(Tau_bond*t) + betaf*np.cos(Tau_bond*t)
    
    # dont forget the normalization and other factors
    return α0, β0  

def get_DD_init_for_fixed_orbit(k_prime,θf,ϕ):
	# calculated the initial conditions of the DD such that, when the QPC hits the bond
	# its state is the same given by thetaf and follows an orbit between 0 a 1 with fixed phi
    # Here we achieve this by shfiting time appropriately
	# k_prime: float. The momentum of the qpc particle

    τ0t = np.arccos(θf) - t*bond_index/(2*J[0]*np.sin(k_prime))
    alpha0 = np.cos(-τ0t)
    beta0 = -1j*np.sin(-τ0t)*np.exp(1j*ϕ)                   
    return alpha0, beta0

def get_DD_init_for_fixed_k(k_prime):
    # calculated the initial conditions of the DD such that, when the QPC hits the bond
    # its state is the same as that of a DD initialized localized in the first site when 
    # the QPC for that case hits the bond with an average momentum k0=pi/2
    # k_prime: float. The momentum of the qpc particle
     
    alpha0 = np.cos( (t*bond_index)/(2*J[0])*(1/np.sin(k_prime) - 1) )
    beta0 = - 1j*np.sin( (t*bond_index)/(2*J[0])*(1/np.sin(k_prime) - 1) )
                        
    return beta0, alpha0


def schmidt_qubit_qpc(c_eigs):
    # calculates the schmidt decomposition between the QPC and qubit for some stae
    # given how we build our hamiltonian, the 0qubit-eigenstates are in the even indices
    # while the 1qubit ones are in the odd indices
    # c_eigs is the state vector

    # arange in MPS form for SVD
    col_1 = np.asarray(c_eigs[0::2]).reshape(-1,1)
    col_2 =  np.asarray(c_eigs[1::2]).reshape(-1,1)
    psi_mat = np.concatenate((col_1, col_2), axis=1)
    U, S, Vh = np.linalg.svd(psi_mat)

    return U, S, Vh

def get_QPC_occupations(Psi):
    # apply the density opperator to the full composite Psi and return the
    # densities a teach of the QPC sites
    occ_list = []

    for j in range(0,L_qpc):
        basis_j = np.zeros(L_qpc)
        basis_j[j] = 1
        # create density operator for the jth qoc lattice
        ketbra = np.outer(basis_j, np.conj(basis_j))
        # tensor with the qubit identity
        Nj_op = np.kron(ketbra, np.eye(2))
        occ_list.append(np.vdot(Psi, Nj_op @ Psi).real)
    return np.asarray(occ_list)

def get_qubit_occupations(Psi):
    # apply the density opperator to the full composite Psi and return the
    # densities a teach of the QPC sites
    occ_list = []

    # create density operator for the 0th qubit state 
    ketbra = np.outer([1,0], np.conj([1,0]))
    # tensor with the qubit identity
    Nj_op = np.kron(np.eye(L_qpc), ketbra)

    return np.vdot(Psi, Nj_op @ Psi).real


def time_evolve(Psi0, Tau, states_, energies_):
    # time evolve a state up psi0 to time Tau with eigenstates states_m for anti-band
    # and states_p for symmetric band
    psi_t = np.zeros(len(Psi0)) + 0j
    for k in range(0,len(energies_)):
        psi_t += np.exp(-1j*Tau*energies_[k])*np.dot( np.conj(states_[:,k]), Psi0)*states_[:,k]

    return psi_t

def get_reduced_density_matrix(Psi,NN):
    # calcualtes the partial trace from a STATEVECTOR Psi
    # NOT from the density matrix
    # Psi: array with the wavefunction
    # NN: integer with the total size of the lattice

    n = (NN-2) # QPC sites
    m = 2 # Double dot sites
    # get density matrix as a sparse array
    ρ = np.outer(Psi, np.conj(Psi))
    # trace out QPC sites and return reduced rho for DD as Quobject
    return np.trace(ρ.reshape(n,m,n,m), axis1=0, axis2=2)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

# create all the combinations of parameters that we want
comb_array = np.array( np.meshgrid(L_qpc_list,Omega_list,t_list,
			J_prime_list,bond_index_list,K0_list,centered_at_list,
			Delta_list,maxt_time_list,N_timepoints_list,ddot_list,
			 phi_list,af_list)).T.reshape(-1, Nparams) 

for simulation_index in tqdm(range(0,np.shape(comb_array)[0]), desc="Iterating Parameters"):

	# initialize current iteration
	parameter_array = comb_array[simulation_index,:]
	L_qpc = int(parameter_array[0])
	Omega = float(parameter_array[1])
	t = float(parameter_array[2])
	J_prime = float(parameter_array[3])
	bond_index = int(parameter_array[4])
	K0 = float(parameter_array[5])
	centered_at = int(parameter_array[6])
	Delta = float(parameter_array[7]) 
	maxt_time = float(parameter_array[8]) 
	N_timepoints = int(parameter_array[9])
	ddot = str(parameter_array[10])
	phi = float(parameter_array[11])
	af = float(parameter_array[12])

	L = L_qpc + 2
	J = np.ones(L_qpc) 
	J[bond_index] = J_prime  
	bf = np.sqrt(1-af**2) # probability of qubit 1 state
	x = np.arange(0,L_qpc)  # latice sites

	print("calculating for: ")
	print("")
	print("L_qpc=", L_qpc,  "maxt_time=",maxt_time, "bond_index=",bond_index, 
		"centered_at=", centered_at, "band_width=",Delta,"K0=",K0, "J_prime=",J_prime,"t=", 
		t, "Omega=", Omega, "dot init=",ddot)
	print("...")
	print(" ")

	# build gaussian wavepacket
	alphas = np.exp(-0.5*(x - centered_at)**2 / (Delta**2)) * np.exp(1j * K0 *(x-centered_at))
	norm_ = np.linalg.norm(alphas)
	alphas = alphas/norm_

	if ddot == "momentum":
	    #this is so we get the same qubit state when the qpc hits the bond
	    a0, b0 = get_DD_init_for_momentum(K0, af, bf, phi)
	    
	elif ddot == "free":
	    a0 = af
	    b0 = bf
	elif ddot == "fixed":
	    a0, b0 = get_DD_init_for_fixed_orbit(K0,af,phi)

	elif ddot == "old":
		a0, b0 = get_DD_init_for_fixed_k(K0)
	else:
	    print("Invalid Initial state for the qubit")

	psi_0 = np.kron(alphas, [a0,b0])

	# build the projection opeartor to the qubit-symmetric sector of the hilbert space
	Id_qpc = np.eye(L_qpc)
	plusket = np.asarray([1/np.sqrt(2),1/np.sqrt(2)])
	p_ketbra = np.outer(plusket,plusket)
	Psym = np.eye(2*L_qpc) - np.kron(Id_qpc, p_ketbra) # tensor product

	# build the momentum values for the bands
	k_single_band = np.arange(1,L_qpc+1)*np.pi/(L_qpc+1)

	# Build the hamiltonian and diagonalize it
	H_matrix, Hdeco = create_hamiltonians(L_qpc,t, bond_index)

	# for the interacting hamiltonian
	energies, eigen_vecs = np.linalg.eig(H_matrix)
	# normalize
	eigen_vecs = eigen_vecs/ np.linalg.norm(eigen_vecs, axis=0)
	# Calculate the non-interacting energies and eigenvectors
	free_energies, free_eigen_vecs = np.linalg.eig(Hdeco)
	# normalize
	free_eigen_vecs = free_eigen_vecs/ np.linalg.norm(free_eigen_vecs, axis=0)
	sorted_indices, over_matrix = sort_by_overlap_matrix(energies, free_eigen_vecs,eigen_vecs)

	# now sort the freee case into bands according to projection
	mindices, pindices = sort_by_projection(free_energies,free_eigen_vecs, Psym)

	free_energies_m, free_states_m, free_energies_p, free_states_p = get_bands(free_energies, free_eigen_vecs,
																			 mindices, pindices)

	# now sort the coupled eigenvectors and values according to max overlaps so they match with the free case
	sorted_e = energies[sorted_indices]
	sorted_vecs = eigen_vecs[:,sorted_indices]
	over_energies_m, over_states_m, over_energies_p, over_states_p = get_bands(sorted_e, sorted_vecs, 
																				mindices, pindices) 

	# time evolve the initial state 
	time_range = np.linspace(0, maxt_time, N_timepoints)
	trajectories = np.zeros((L_qpc, len(time_range)))
	qubit_traj = []
	rho_list = []
	St = []

	i = 0
	for Tau in time_range:
	    psi_t = time_evolve(psi_0, Tau,eigen_vecs, energies)
	    occupations = get_QPC_occupations(psi_t)
	    qubit_traj.append(get_qubit_occupations(psi_t))
	    # reduced density matrix 
	    rhot = get_reduced_density_matrix(psi_t,L_qpc+2)
	    # entropy from schmidt
	    """
	    U, S, Vh = schmidt_qubit_qpc(psi_t)
	    schmis = S**2
	    """
	    # save in arrays
	    trajectories[:,i] = occupations
	    rho_list.append(rhot)
	    # St.append(-1*np.sum(schmis*np.log(schmis+1e-17))) # avoid log(0)
	    St.append(entropy_vn(Qobj(rhot), sparse=False) )

	    i += 1

	# Save results
	param_dict = {"L_qpc": L_qpc, "Omega": Omega, "t":t ,"J":J[0] ,"Jp": J_prime, "bond_index" : bond_index, 
	              "K0": K0, "X0":centered_at, "Spread":Delta, "maxt_time": maxt_time, 
	              "del_tau":time_range[1]-time_range[0], "qubit_init":ddot,  "Re_qubit_0":np.real(a0), 
	              "Im_qubit_0":np.imag(a0), "Re_qubit_1":np.real(b0), "Im_qubit_1":np.imag(b0), "phi":phi,
	              "alfabond": af }

	file_name = "exact_L{}_J{}_t{}_om{}_Del{}_xo{}_k{:.4f}_bindex{}_maxtau{:.3f}_tstep{:.3f}_alpha{:.3f}_beta{:.3f}_phi{}_alpha_bond{:.3f}_qinit{}.h5".format(
	                                    L_qpc, J_prime, t, Omega, Delta,centered_at , K0, bond_index,maxt_time,
	                                    time_range[1]-time_range[0], np.abs(a0)**2, np.abs(b0)**2, phi, af,ddot)

	results_file = h5py.File(data_route+file_name,'w')
	# save parameters and maybe other meta data
	grp = results_file.create_group("metadata")
	grp.create_dataset("parameters", data=json.dumps(param_dict))

	# save the quantities that we are interested in 
	grp = results_file.create_group("results")
	grp.create_dataset("time", data=time_range )
	grp.create_dataset("trajectories", data=trajectories)
	grp.create_dataset("d0_density", data=np.asarray(qubit_traj))
	grp.create_dataset("qubit_rho", data=rho_list)
	grp.create_dataset("Entropy", data=St)

	results_file.close()
	gc.collect()




