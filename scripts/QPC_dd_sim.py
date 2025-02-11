import numpy as np
from scipy import sparse
from qutip import  *
import os
import sys
import h5py
import json

from tqdm import tqdm


# add path to project dir so we can include custom modules from src

# import our cusstom module
from qutip_tools import *

# Finds the time evolution of a QPC coupled to a double dot (dd) by exact 
# diagonalization using qutip. The raw data is save as the trajectories and the
# time evolution of the wavefunciton at the end. 

# --------------------------------
# PARAMETERS
# --------------------------------

# location where the raw data is saved
data_route = "../data/sims/L=20/"
# data_route = "/scratch/scc/santiago.salazar-jaramillo/hoping_tensors/data/sims/L=20/"
# it is important to have them as lists

L_qpc_list = [16]  # lenth of the QPC chain
L_list = [L_qpc_list[0]+2] # QPC times double dot 
max_t_list = [9] # maximum time
tsteps_list = [300] # number of time steps
bond_index_list = int(L_qpc_list[0]/2) # dangling bond between bond_index and bond_index+1
centered_at_list = [0] # initial QPC position of wavepacket
band_width_list = [0.5] #[0.5, 2.0] # width of the gaussian wave packet
K0_list = [np.pi/2] #[np.pi/8, np.pi/6,np.pi/4, 5*np.pi/16, 6*np.pi/16, 7*np.pi/16 ,np.pi/2] # Initial velocity of the wavepacket
J_prime_list = [1.0] # contact to double dot
t_list = [0.2]#[0.0, 0.2, 0.4, 0.9]# hopping between quantum dots 
Omega_list = [0.0] #[0.0, 0.1, 0.3 ,0.5, 0.7]  # coupling between dot 1 and QPC
ddot0_list = ['fixed'] #["second","fixed"] # # can be first (loc in 1st site), second (loc in 2nd) or fixed (fixed by K0)
# this is just to get the number of params for the combinations later
Nparams = 12

# --------------------------------
# Functions 
# --------------------------------

def gen_gauss_init(l0, sigma, Nsites, k0=0):
    # creates a gaussian initial condition centerd on l0 with bandwidth sigma for Nsites
    # and initial velocity k0

    x = np.asarray(range(0,Nsites))
    coefs = ((np.sqrt(np.pi)*sigma)**(-0.5))*np.exp(-0.5*(x-l0)**2/(sigma**2) )*np.exp(1j*k0*(x-l0))
    
    # normalize
    mag = np.dot(np.conjugate(coefs),coefs)
    coefs = coefs/np.sqrt(mag)
    
    return coefs    

def get_DD_init_for_fixed_k(k_prime):
    # calculated the initial conditions of the DD such that, when the QPC hits the bond
    # its state is the same as that of a DD initialized localized in the first site when 
    # the QPC for that case hits the bond with an average momentum k0=pi/2
    # k_prime: float. The momentum of the qpc particle
     
    alpha0 = np.cos( (t*bond_index)/(2*J[0])*(1/np.sin(k_prime) - 1) )
    beta0 = - 1j*np.sin( (t*bond_index)/(2*J[0])*(1/np.sin(k_prime) - 1) )
                        
    return alpha0, beta0

def get_qpc_H(op_list, Nsites, Nqpc,jcouple):
        # create the Hamiltonian for the QPC where Nsites includes the double dot
    # and Nqpc only has the qpc site
    ident_tensor = tensor([identity(2)]*(Nsites)) 
    H = 0*ident_tensor

    for site_j in range(0,Nqpc-1):
        H += -jcouple[site_j]*(op_list[site_j].dag()*op_list[site_j+1]+
                                   op_list[site_j+1].dag()*op_list[site_j])
    return H 

def gen_QPC_dot_basis(L_QPC, Center_index, Band_w, Kinit, DD0):
    # Combines the 1particle bassi of the QPC and the dot to get the full psi

    # L_QPC: integer, length of qpc lattice
    # Center_index: integer, indicates the lattice site where QPC is initialized
    # Band_w: float, band width of the gaussian qave packet in the qpc
    # Kinit: float, group velocity of the gaussian wave packet
    # DD0: string, tells dot initial condition either "first" or "second"
    
    # create the 1 particle basis and the coeficients for the initial state
    str_list, basis_list = get_1p_basis(L_QPC)
  
    # build the initial condition for the QPC
    qpc_init = gen_gauss_init(Center_index, Band_w, L_QPC, Kinit)
    psi_qpc = [qpc_init[j]*basis_list[j] for j in range(0,len(qpc_init))]

    # make sure these stay as sparse matrices
    b1 = basis(2,0)
    b1.data = data.to(data.CSR, b1.data)
    
    b2 = basis(2,1)
    b2.data = data.to(data.CSR, b2.data)

    # create the dot basis
    dot_basis = [tensor(b1,b2), tensor(b2,b1)]
    
    # build the initial condition for the dot
    if DD0 == "first": 
        dot_init = [1.0, 0.0]
    elif DD0 == "second": 
        dot_init = [0.0, 1.0]
    elif DD0 == "fixed":
        a0, b0 = get_DD_init_for_fixed_k(Kinit)
        dot_init = [a0,b0]
    else:
        print("Invalid initial condition for the double dot")

    psi_dot = [dot_basis[j]*dot_init[j] for j in range(0,len(dot_basis))]
    # assume dot initial state completely independent from QPC init state so we can factorize the probas
    full_basis = []
    # combine them 
    for i in range(0,len(basis_list)):
        for j in range(0, len(dot_basis)):
            full_basis.append(tensor([psi_qpc[i], psi_dot[j]]))

    # state correspond to particle in qpc all the way to the left and particle on left dot
    Psi0 = np.sum(full_basis)
    Psi0 = Psi0.unit()   
    
    return Psi0, qpc_init

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

def get_entanglement(States, L ,tskip=5):
    # calculates several entanglement measures
    # States: list of Quobj containing the time evolution of the wavefunction
    # tskip: tells how many in between times to skip for faster computation
    purity_list = []
    entropy_list = []
    corr_rho_list = []
    theta_list = []
    phi_list = []

    # skip some times otherwise its too heavy
    state_arr = States[0::tskip]
    for ti in range(0,len(state_arr)):
        # DD reduced density matrix
        # rho_DD = get_partial_trace(state_arr[ti], L)
        rho = state_arr[ti]*state_arr[ti].dag()
        rho_DD = rho.ptrace(sel=[L-2,L-1])

        # cut redundant degrees for blochsphere calculation
        r = Qobj(rho_DD[1:-1,1:-1])
        # mixed state bloch sphere representation. Check page 34 of my notes for this 
        Sig_matrix = 2*r - identity(2)
        theta_p = np.arccos(Sig_matrix[0,0])
        phi_p = -np.arccos( 0.5*(Sig_matrix[1,0]+Sig_matrix[0,1])/np.sin(theta_p) )

        # purity
        purity_list.append((rho_DD**2).tr())
        entropy_list.append(entropy_vn(rho_DD, sparse=False))
        theta_list.append(theta_p)
        phi_list.append(phi_p)
        
    return purity_list, entropy_list, theta_list, phi_list, tskip

# ---------------------------
# MAIN 
# ----------------------------

# create all the combinations of parameters that we want
comb_array = np.array( np.meshgrid(L_qpc_list,L_list, max_t_list, tsteps_list, bond_index_list,  
                                    centered_at_list, band_width_list, K0_list, J_prime_list, 
                                    t_list, Omega_list, ddot0_list)).T.reshape(-1, Nparams) 

for simulation_index in tqdm(range(0,np.shape(comb_array)[0]), desc="Iterating Parameters"):

    parameter_array = comb_array[simulation_index,:]
    # initialize the parameters of the current iteration
    L_qpc = int(parameter_array[0])
    L = int(parameter_array[1])
    max_t = float(parameter_array[2])
    tsteps = int(parameter_array[3])
    bond_index = int(parameter_array[4])
    centered_at = int(parameter_array[5])
    band_width = float(parameter_array[6])
    K0 = float(parameter_array[7])
    J_prime = float(parameter_array[8])
    t = float(parameter_array[9])
    Omega = float(parameter_array[10])
    ddot = parameter_array[11]

    print("calculating for: ")
    print("")
    print("L_qpc=", L_qpc, "L=", L, "max_t=",max_t, "tsteps=",tsteps, "bond_index=",bond_index, "centered_at=", centered_at, 
        "band_width=",band_width,"K0=",K0,"J_prime=",J_prime,"t=", t, "Omega=", Omega, "dot init=",ddot)
    print("...")
    print(" ")

    J = np.ones(L_qpc) # QPC hopping
    # this means that I am putting the dangling bond between sites int(L_qpc/2) and int(L_qpc/2)+1
    # where the interaction to the double dot is also located
    J[bond_index] = J_prime  

    # create the QPCxDot basis
    psi0, qpc_init = gen_QPC_dot_basis(L_qpc, centered_at, band_width, K0,ddot)
    # create the fermion operator list
    c_list = [fdestroy(L,i) for i in range(0,L)]

    # create the hamiltonian start with qpc
    H_QC = get_qpc_H(c_list, L ,L_qpc,J)

    # double dot H
    Hdot = -t*(c_list[-1].dag()*c_list[-2] + c_list[-2].dag()*c_list[-1])

    # interaction H
    Hint =  Omega*c_list[-2].dag()*c_list[-2]*( c_list[bond_index].dag()*c_list[bond_index+1] +
                                              c_list[bond_index+1].dag()*c_list[bond_index] )

    H = H_QC + Hdot  + Hint

    # get the operators needed for lindbladian 
    # hrtr we don't really want dephasing (gamma=0) but just put it there for now
    collapse_ops, expect_ops = create_lindblad_op(L, c_list, 0.0)
    # add the energy to also track it
    expect_ops.append(H)

    # solve the schroedinger equation
    times = np.linspace(0.0, max_t, tsteps)
    # solve for the operators
    result = sesolve(H, psi0, times, e_ops=expect_ops,options={"store_states": True})

    # calculate sum of occupations
    # exclude the sites at Lp/2 and Lp/2 +1 where the bond is located
    n_left = np.sum(result.expect[:bond_index], axis=0)
    # the minus 3 is because we  leave out the energy and dot occupations
    n_right = np.sum(result.expect[bond_index+2:-3], axis=0)
    # occupation in the bond
    n_bond = result.expect[int(bond_index)] + result.expect[bond_index+1] 

    # Calculate the entropy
    time_skip = 20 
    purity_list , entropy_list, theta_list, phi_list, tskip = get_entanglement(result.states, L_qpc+2 ,tskip=time_skip)

    # save results to hdf5 file
    file_name = "res_L{}_maxtim{}_bw{}_k{:.4f}_jp{}_t{}_om{}_dd0{}.hdf5".format(L_qpc, max_t, band_width, 
                                                                      K0, J_prime, t, Omega,ddot)

    param_dict = {"L_qpc": L_qpc, "max_time": max_t,"tsteps": tsteps,"bond_index": bond_index, 
                  "band_width": band_width,"k0":K0, "J_prime":J_prime , "t": t, "Omega": Omega,
                  "ddot0":ddot, "centered_at":centered_at, "entropy_t_skip": time_skip }

    results_file = h5py.File(data_route+file_name,'w')
    # save parameters and maybe other meta data
    grp = results_file.create_group("metadata")
    grp.create_dataset("parameters", data=json.dumps(param_dict))

    # save the quantities that we are interested in 
    grp = results_file.create_group("results")
    grp.create_dataset("time", data=times )
    grp.create_dataset("d1_density", data=result.expect[-3])
    grp.create_dataset("d2_density", data=result.expect[-2])
    grp.create_dataset("energy", data=result.expect[-1])
    grp.create_dataset("QPC_left_density", data=n_left)
    grp.create_dataset("QPC_right_density", data=n_right)
    grp.create_dataset("QPC_bond_density", data=n_bond)
    grp.create_dataset("trajectories", data=result.expect[:-3])
    grp.create_dataset("dot_purity", data=purity_list)
    grp.create_dataset("dot_VN_entropy", data=entropy_list)
    grp.create_dataset("dot_bloch_theta", data=theta_list)
    grp.create_dataset("dot_bloch_phi", data=phi_list)
    
    results_file.close()

    """# now save the wavefunction
                file_name = "psi_L{}_maxtim{}_bw{}_k{:.4f}_jp{}_t{}_om{}_dd0{}".format(L_qpc, max_t, band_width, 
                                                                                  K0, J_prime, t, Omega,ddot)
                qsave(result.states, data_route+"wavefunctions/"+file_name)
            """



    
