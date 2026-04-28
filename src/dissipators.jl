raw"""
    get_local_density_kraus_operators(site, gamma, dt)

Returns the Kraus operators for a local density dissipative Lindblad term
D_i = 2n_i \rho n_i^\dag - \{n_i^\dag, n_i, \rho\}.

The Kraus representation of the map is 
\rho \rightarrow exp(dt*D_i) \rho = K_1 \rho K_1^\dag + K_2 \rho K_2^\dag 
with
K_1 = id + n_i * (exp(-γ dt) - 1)
and
K_2 = √(1-exp(-2γdt)) * n_i.
Note that to arrive at this representation you need that n_i = n_i^† and 
n_i^2 = n_i.
"""
function get_local_density_kraus_operators(site, gamma, dt)
    dens = replaceprime(prime(op("S+", site)) * op("S-", site), 2, 1)
    K_1 = delta(site, prime(site)) + dens * (exp(-gamma * dt) - 1)
    K_2 = sqrt(1 - exp(-2 * gamma * dt)) * dens
    return [K_1, K_2]
end


raw"""
    get_local_dissipative_kraus_operators(site, gamma, dt)

Returns the Kraus operators for a local dissipative Lindblad term
D_i = 2a_i \rho a_i^\dag - \{n_i, \rho\}.

The Kraus representation of the map is 
\rho \rightarrow exp(dt*D_i) \rho = K_1 \rho K_1^\dag + K_2 \rho K_2^\dag 
with
K_1 = id + n_i * (exp(-γ dt) - 1)
and
K_2 = √(1-exp(-2γdt)) * a_i.
"""
function get_local_dissipative_kraus_operators(site, gamma, dt)
    dens = replaceprime(prime(op("S+", site)) * op("S-", site), 2, 1)
    K_1 = delta(site, prime(site)) + dens * (exp(-gamma * dt) - 1)
    K_2 = sqrt(1 - exp(-2 * gamma * dt)) * op("S-", site)
    return [K_1, K_2]
end


raw"""
    get_generalized_amplitude_damping(site, mu, gamma, dt)

Returns the Kraus operators for a generalized amplitude damping (spontaneous
    emission and spontaneous excitation at the same time). See arXiv:1902.00967v2 
    for details.

The map has 4 Kraus ops (2 for emission, two for excitation), and mu sets the 
bias between the two (the difference in probability for the two to occur).
        mu > 0: excitation more probable than emission
"""
function get_generalized_amplitude_damping(site, mu, gamma, dt)
    full = replaceprime(prime(op("S+", site)) * op("S-", site), 2, 1)
    empty = replaceprime(prime(op("S-", site)) * op("S+", site), 2, 1)

    # spontaneous emission
    K_1 = sqrt((1-mu)/2) * (delta(site, prime(site)) + full * (exp(-gamma * dt) - 1))
    K_2 = sqrt((1-mu)/2) * sqrt(1 - exp(-2 * gamma * dt)) * op("S-", site)
    if mu == -1
        return [K_1, K_2]
    end
    # spontaneous excitation
    K_3 = sqrt((1+mu)/2) * (delta(site, prime(site)) + empty * (exp(-gamma * dt) - 1))
    K_4 = sqrt((1+mu)/2) * sqrt(1 - exp(-2 * gamma * dt)) * op("S+", site)
    if mu == 1
        return [K_3, K_4]
    end
    return [K_1, K_2, K_3, K_4]
end


"""
    get_local_density_kraus_hole_operators(site, gamma, dt)

Returns the hole Kraus operators for a local density dissipative Lindblad term, 
see eq. (18) from PRR 2, 043052 (2020)

pre_coeffs is the precoefficient given by the Wilson transformation for the 
operators. If nothing is given, 1 is chosen such that nothing changes
"""

function get_local_density_kraus_hole_operators(site, gamma, dt; kwargs...)
    pre_coeffs = get(kwargs, :pre_coeffs, 1)
    dens = pre_coeffs*conj(pre_coeffs)*replaceprime(prime(op("Cdag", site)) * op("C", site), 2, 1)
    dag_dens = pre_coeffs*conj(pre_coeffs)*replaceprime(prime(op("C", site)) * op("Cdag", site), 2, 1)
    K_1 = dag_dens + dens *exp(-gamma * dt / 2.0)
    K_2 = conj(pre_coeffs)*sqrt(1 - exp(- gamma * dt)) * op("C", site)
    return [K_1, K_2]
end

"""
    get_local_density_kraus_particle_operators(site, gamma, dt)

Returns the particle Kraus operators for a local density dissipative Lindblad term, 
see eq. (18) from PRR 2, 043052 (2020)
"""
function get_local_density_kraus_particle_operators(site, gamma, dt; kwargs...)
    pre_coeffs = get(kwargs, :pre_coeffs, 1)
    dens = pre_coeffs*conj(pre_coeffs)*replaceprime(prime(op("Cdag", site)) * op("C", site), 2, 1)
    dag_dens = pre_coeffs*conj(pre_coeffs)*replaceprime(prime(op("C", site)) * op("Cdag", site), 2, 1)
    K_1 = dens + dag_dens *exp(-gamma * dt / 2.0)
    K_2 = pre_coeffs*sqrt(1 - exp(- gamma * dt)) * op("Cdag", site)
    return [K_1, K_2]
end


"""
   create_dissipative_kraus_p_h_gates_fermions(sites, impurity_sites_number, dt, 
                                   gamma, nrg_h_coeffs, nrg_p_coeffs; kwargs...)

Returns the particle Kraus operators for a local density dissipative Lindblad term, 
see eq. (18) from PRR 2, 043052 (2020)
"""
function create_dissipative_kraus_p_h_gates_fermions(sites, impurity_sites_number::Array{Int64, 1},
                                                     dt, gamma; kwargs...)
    #ToDo: implement error message if the arrays nrg_p_coeffs and nrg_h_coeffsare note given
    dt = Float64(dt)
    gamma = Float64(gamma)
    
    nrg_p_coeffs = get(kwargs, :nrg_p_coeffs, "")
    nrg_h_coeffs = get(kwargs, :nrg_h_coeffs, "")

    n_sites = length(sites)
    n_imp_sites = length(impurity_sites_number)
    n_chain_sites_p = length(nrg_p_coeffs) #length of particle chain
    #### give range of hole and particles sites
    p_sites = 1:n_chain_sites_p
    h_sites = n_chain_sites_p+n_imp_sites+1:n_sites
    K_array = [] #add particle Kraus operators 
    for j in p_sites
       ops = get_local_density_kraus_particle_operators(sites[j], gamma, dt; 
                                                        pre_coeffs = nrg_p_coeffs[end+1-j])
        #for the Kraus operator to be applied correctly in do_evolution_sweep!
        #therefore, the [ops] entry. This is
       append!(K_array , [ops])
    end

    for (i,j) in enumerate(h_sites)
       ops = get_local_density_kraus_hole_operators(sites[j], gamma, dt;
                                                    pre_coeffs = nrg_h_coeffs[i])
       append!(K_array, [ops])
    end
    gate_sites = vcat(p_sites, h_sites) #inidices of sites where operators act
    return K_array, gate_sites
end