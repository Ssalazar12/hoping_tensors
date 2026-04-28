"""
    create_gates_from_single_site_ampo(single_site_ampo_array, sites, dt)

Creates single-site unitary time evolution gates for all sites,
with evolution by time step dt.
"""

function create_gates_from_single_site_ampo(single_site_ampo::AutoMPO, sites, dt)
    gates = []
    for site in sites
        #ITensors site indicces must be arrays, therefore we define [sites[site]]
        H = MPO(single_site_ampo, [sites[site]]) 
        push!(gates, exp(-1im * dt .* H[1]))
    end
    return gates
end

function create_gates_from_single_site_ampo(single_site_ampo_array::Vector, sites, dt::Float64)
    gates = []
    for (site,ampo) in enumerate(single_site_ampo_array)
        #ITensors site indicces must be arrays, therefore we define [sites[site]]
        H = MPO(ampo, [sites[site]])
        push!(gates, exp(-1im * dt .* H[1]))
    end
    return gates
end

"""
    create_gates_from_two_site_ampo(two_site_ampo_array, sites, dt)

Creates two-site unitary time evolution gates for all sites,
with evolution by time step dt.
"""
function create_gates_from_two_site_ampo(two_site_ampo::AutoMPO, sites, dt)
    gates = []
    for site = 1:length(sites)-1
        H = MPO(two_site_ampo, sites[site:site+1])
        push!(gates, exp(-1im * dt .* H[1] * H[2]))
    end
    return gates
end

function create_gates_from_two_site_ampo(two_site_ampo_array, sites, dt)
    gates = []
    for (site, ampo) in enumerate(two_site_ampo_array)
        H = MPO(ampo, sites[site:site+1])
        push!(gates, exp(-1im * dt .* H[1] * H[2]))
    end
    return gates
end


"""
    apply_gate_to_state!(state, gate, site_number; kwargs...)

Apply a gate to a state. So far, only single-sites and two-site gates can be applied to the state.
kwargs: maxdim and cutoff variables for svd in  apply_two_site_op_to_state!
"""
function apply_gate_to_state!(state, gate, site_number; kwargs...)
    if number_of_inds(gate) == 1
        return apply_single_site_op_to_state!(state, gate, site_number)
    elseif number_of_inds(gate) == 2
        return apply_two_site_op_to_state!(state, gate, site_number; kwargs...)
    else
        throw(ArgumentError("can only apply single-site and two-site operators so far"))
    end
end


"""
    apply_kraus_operators_to_state!(state, gate, site_number)

Apply single site Kraus operators to a state. `gate` is an array of single site
Kraus operators, [E_1, ..., E_n]. The resulting state will be
E_1 * state * E_1^dag + ... + E_n * state * E_n^dag.
"""
function apply_kraus_operators_to_state!(state, gate, site_number)
    if !all([number_of_inds(op) == 1 for op in gate])
        throw(ArgumentError( "can only apply single-site Kraus operators so far"))
    end
    site_tensor = deepcopy(state[site_number])
    new_site_tensor = sum([op * (site_tensor * prime(dag(op))) for op in gate])
    prime!(new_site_tensor, -1, "Site")
    state[site_number] = new_site_tensor
    return nothing
end


"""
    apply_hadamard!(state::MPS, site_number::Int)

Apply the Hadamard gate on a certain site of a state represented as MPS.
"""
function apply_hadamard!(state::MPS, site_number::Int)
    siteind = siteinds(state)[site_number]
    Hadamard = get_hadamard(siteind)
    state[site_number] = Hadamard * state[site_number]
    noprime!(state)
    return state
end


"""
    get_hadamard(ind)

Get the Hadamard gate as ITensor. The Hadamard mixes the 0 and 1 states of a 
two-level system according to:
[[1  1],
 [1  -1]]/sqrt(2).
"""
function get_hadamard(ind)
    if ind.space != 2
        throw(ArgumentError("Hadamard only defined for two level index!"))
    end
    pind = prime(ind)
    Hadamard = ITensor(ind, pind)
    Hadamard[ind=>1, pind=>1] = 1/sqrt(2)
    Hadamard[ind=>1, pind=>2] = 1/sqrt(2)
    Hadamard[ind=>2, pind=>1] = 1/sqrt(2)
    Hadamard[ind=>2, pind=>2] = -1/sqrt(2)
    return Hadamard
end
