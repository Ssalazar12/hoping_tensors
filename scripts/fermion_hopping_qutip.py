import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

from qutip import  *

# PARAMS --------------------------------------------

L = 6 # Num of sites
D = 2 # local hilbert space dimension
gamma = 0.2 # the dephasing term for the collapse operators
max_t = 40 # maximum time
tsteps = 500 # number of time steps
J = np.ones(L) # interaction
J_contact = 1.0
# Here we could place a a point contact if we want
J[L-4] = J_contact
J[L-3] = J_contact
collapse_op = "number" # either "number" or "ladder"

centered_in = 4 # which site the gaussian wave packet is centered at
band_width = 0.5 # bandwidth og wave packet. Higher number means less localized
b = destroy(2) # annihilation operator
K0 = 0.01 # initial velocity of the wave packet


# FUNCTIONS --------------------------------------------

def create_fermion_op(Nsites):
    # create fermions from quspin bosons and JW strings
    C_list = []
    for site_j in range(0,Nsites):
        # create the string
        JW_string = [identity(2)]*Nsites
        for l in range(0,site_j):
            JW_string[l] = (1j*np.pi*b.dag()*b).expm()
        JW_string[site_j] = b
        # tensor product the string to get the new fermion operator at site_j
        C_list.append(tensor(JW_string))
    return C_list

def create_lindblad_op(Nsites, operator_list ,collapse_type):
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

def gen_gauss_init(l0, sigma, Nsites, k0=1):
    # creates a gaussian initial condition centerd on l0 with bandwidth sigma

    # coefs = [np.exp(-(1/2*sigma)*(l-l0)**2) for l in range(0,Nsites)]

    x = np.asarray(range(0,Nsites))
    coefs = ((np.sqrt(np.pi)*sigma)**(-0.5))*np.exp(-0.5*(x-l0)**2/(sigma)**2 )*np.exp(1j*k0*(x-l0))
    
    return coefs
    
    
def gen_psi0(init_coefs, Nsites):
    # creates the initial wave function from the init_coefs list
    # and the one particle basis vectors
    # create the density matrix from ONE particle basis states
    # list holding all possible 1-particle states
    basis_list = []
    
    for site_j in range(0,Nsites):
        # create emty sites
        site_vectors = [basis(2, 0)]*Nsites
        # flip up one exittation
        site_vectors[site_j] = basis(2, 1)
        basis_list.append(tensor(site_vectors))
        
        """basis_vector = [0]*Nsites
        # flip one particle
        basis_vector[site_j] = 1
        basis_ket = ket(basis_vector, dim=2)
        basis_list.append(basis_ket)
        """ 
    # create the initial state
    Psi0 = np.sum([init_coefs[j]*basis_list[j] for j in range(0,Nsites)])
    Psi0 = Psi0.unit()
    return Psi0
    

# MAIN --------------------------------------------

# initialize initial gaussian state
initial_conditions = gen_gauss_init(centered_in,band_width, L,k0=K0)
psi0 = gen_psi0(initial_conditions, L)
rho = ket2dm(psi0)

# create fermion operators
c_list = create_fermion_op(L)

# create the hamiltonian in terms of the fermion operators
ident_tensor = tensor([identity(D)]*L) 
H = 0*ident_tensor

for site_j in range(0,L-1):
    H += -0.5*J[site_j]*(c_list[site_j].dag()*c_list[site_j+1]+c_list[site_j+1].dag()*c_list[site_j] )

collapse_ops, expect_ops = create_lindblad_op(L, c_list ,collapse_op)
# add the energy to also track it
expect_ops.append(H)

times = np.linspace(0.0, max_t, tsteps)
result = mesolve(H, rho, times, c_ops=collapse_ops , e_ops=expect_ops)

print(result)

# PLOTTING ---------------------------------------

fig, ax = plt.subplots(1,1, figsize=(7,5))

# plot
for i in range(0,len(result.expect)-1):
    ax.plot(times, result.expect[i], linewidth=2.0)
    
# get the total particle number
# get the sum of the local densities
n_tot = np.sum(result.expect, axis=0)
ax.plot(times, n_tot, c='black', linewidth=2.0)

ax.set_xlabel("times")
ax.set_ylabel("local densities")

plt.savefig("../plots/densities_L={}_Jc={}_bw={}.pdf".format(L,J_contact,band_width))

fig, ax = plt.subplots(1,1, figsize=(7,5))

ax.plot(times, result.expect[-1], linewidth=2.0)

ax.set_xlabel("times")
ax.set_ylabel("Energy")



xt = np.linspace(0,max_t,5)
# plot the occupations as a heatmap with lattice site in the y axis and time on the x
ax = sns.heatmap(result.expect[:-1])
ax.set_xlabel("steps")
ax.set_ylabel("densities")

plt.savefig("../plots/trajectory_L={}_Jc={}_bw={}.png".format(L,J_contact,band_width))




