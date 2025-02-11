import numpy as np
from qutip import  *

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


