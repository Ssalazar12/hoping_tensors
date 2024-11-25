import numpy as np
from qutip import  *
import os
import sys
import h5py
import json

# add path to project dir so we can include custom modules from src
path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))

sys.path.insert(0, parent_path)

# import our cussomt module
from src.qutip_tools import *

# --------------------------------
# PARAMETERS
# --------------------------------

# location where the raw data is saved
data_route = "../data/sims/"

L_qpc = 10  # lenth of the QPC chain
L = L_qpc+2 # QPC times double dot 
max_t = 10 # maximum time
tsteps = 200 # number of time steps
bond_index = int(L_qpc/2) # dangling bond between bond_index and bond_index+1
gamma = 0.0
centered_at = 0 # initial QPC position of wavepacket
band_width = 0.8 # width of the gaussian wave packet
K0 = 0.01 # Initial velocity of the wavepacket
J_prime = 0.5 # contact to double dot
t = 1.0 # hopping between quantum dots 
Omega = 0.0 # coupling between dot 1 and QPC

# dot energies not yet implemented
E1 = 0.0
E2 = 0.0

# --------------------------------
# Functions 
# --------------------------------

def gen_gauss_init(l0, sigma, Nsites, k0=0):
    # creates a gaussian initial condition centerd on l0 with bandwidth sigma for Nsites
    # and initial velocity k0

    x = np.asarray(range(0,Nsites))
    coefs = ((np.sqrt(np.pi)*sigma)**(-0.5))*np.exp(-0.5*(x-l0)**2/(sigma)**2 )*np.exp(1j*k0*(x-l0))
    
    # normalize
    mag = np.dot(np.conjugate(coefs),coefs)
    coefs = coefs/np.sqrt(mag)
    
    return coefs    

def get_qpc_H(op_list, Nsites, Nqpc,jcouple):
        # create the Hamiltonian for the QPC where Nsites includes the double dot
    # and Nqpc only has the qpc site
    ident_tensor = tensor([identity(2)]*(Nsites)) 
    H = 0*ident_tensor

    for site_j in range(0,Nqpc-1):
        H += -jcouple[site_j]*(op_list[site_j].dag()*op_list[site_j+1]+
                                   op_list[site_j+1].dag()*op_list[site_j])
    return H 

# ---------------------------
# MAIN 
# ----------------------------

epsilon = np.zeros(L_qpc) # energies of the QPC chains 
epsilon[:bond_index] = 0.0 # source energies
epsilon[bond_index:] = 0.0 # drain energies
J = np.ones(L_qpc) # QPC hopping
# this means that I am putting the dangling bond between sites int(L_qpc/2) and int(L_qpc/2)+1
# where the interaction to the double dot is also located
J[bond_index] = J_prime  

# create the 1 particle basis and the coeficients for the initial state
str_list, basis_list = get_1p_basis(L_qpc)
# build the initial condition for the QPC
qpc_init = gen_gauss_init(centered_at, band_width, L_qpc, k0=K0)

psi_qpc = [qpc_init[j]*basis_list[j] for j in range(0,len(qpc_init))]

# create the dot basis
dot_basis = [tensor(basis(2,0),basis(2,1)), tensor(basis(2,1),basis(2,0))]
# build the initial condition for the dotQ
dot_init = [0.0, 0.1]
psi_dot = [dot_basis[j]*dot_init[j] for j in range(0,len(dot_basis))]

# assume dot initial state completely independent from QPC init state so we can factorize the probas
full_basis = []
# combine them 
for i in range(0,len(basis_list)):
    for j in range(0, len(dot_basis)):
        full_basis.append(tensor([psi_qpc[i], psi_dot[j]]))
      
# state correspond to particle in qpc all the way to the left and particle on left dot
psi0 = np.sum(full_basis)
psi0 = psi0.unit()   

# normalize       
rho = ket2dm(psi0) # initial density matrix
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
# hrtr we don't really want dephasing but just put it there for now
collapse_ops, expect_ops = create_lindblad_op(L, c_list, gamma)
# add the energy to also track it
expect_ops.append(H)


# solve the schroedinger equation
times = np.linspace(0.0, max_t, tsteps)
result = sesolve(H, psi0, times, e_ops=expect_ops)

# calculate sum of occupations
# exclude the sites at Lp/2 and Lp/2 +1 where the bond is located
n_left = np.sum(result.expect[:bond_index], axis=0)
# the minus 3 is because we  leave out the energy and dot occupations
n_right = np.sum(result.expect[bond_index+2:-3], axis=0)
# occupation in the bond
n_bond = result.expect[int(bond_index)] + result.expect[bond_index+1] 

# save results to hdf5 file
file_name = "res_L{}_maxtim{}_bw{}_k{}_jp{}_t{}_om{}.hdf5".format(L_qpc, max_t, band_width, 
                                                                  K0, J_prime, t, Omega)
param_dict = {"L_qpc": L_qpc, "max_time": max_t,"tsteps": tsteps,"bond_index": bond_index, 
              "band_width": band_width,"g_velocity":K0, "J_prime":J_prime , "t": t, "Omega": Omega }

results_file = h5py.File(data_route+file_name,'w')
# save parameters and maybe other meta data
grp = results_file.create_group("metadata")
grp.create_dataset("parameters", data=json.dumps(param_dict))

# save the quantities that we are interested in 
grp = results_file.create_group("results")
grp.create_dataset("time", data=times)
grp.create_dataset("d1_density", data=result.expect[-3])
grp.create_dataset("d2_density", data=result.expect[-2])
grp.create_dataset("QPC_left_density", data=n_left)
grp.create_dataset("QPC_right_density", data=n_right)
grp.create_dataset("QPC_bond_density", data=n_bond)

results_file.close()




    