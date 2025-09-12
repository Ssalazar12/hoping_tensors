logger = getlogger(@__MODULE__)

"""
    measure_operator(operator, state::MPS)
Measure an operator for a given state.
"""
function measure_operator(operator::ITensor, state::MPS)
    # using the intersection of the operator's and the state's indices we can determine where to
    # orthogonalize the state and to apply the operator
    common_indices = intersect(siteinds(state), inds(operator))
    if length(common_indices) == 1
        common_index = common_indices[1]
        # for orthogonalization we need the actual site number
        site_number = get_site_number(common_index)
        to_measure = deepcopy(state)
        orthogonalize!(to_measure, site_number)
        res = scalar(to_measure[site_number] * operator * dag(prime(to_measure[site_number], common_index)))
    elseif length(common_indices) == 2
        to_measure = deepcopy(state)
        site_numbers = get_site_number.(common_indices)
        orthogonalize!(to_measure, site_numbers[1])
        local_tensor = to_measure[site_numbers[1]] * to_measure[site_numbers[2]]
        res = scalar(local_tensor * operator * dag(prime(local_tensor, common_indices)))
    else
        throw(ArgumentError("can only measure single site and two site operators!"))
    end
    return res
end


"""
    measure_operator(operator, state::MPO)

Measure an operator for a given state in density matrix representation.
"""
function measure_operator(operator::ITensor, state::MPO)
    # using the intersection of the operator's and the state's indices we can determine where to
    # orthogonalize the state and to apply the operator
    #ToDo: implement more elegant way to calculate siteinds
    state_site_inds = [siteinds(state; plev = 0)[i][1] for i in 1:length(state)]
    common_indices = intersect(state_site_inds, inds(operator))
    if length(common_indices) == 1
        common_index = common_indices[1]
        # for orthogonalization we need the actual site number
        site_number = get_site_number(common_index)
        to_measure = deepcopy(state)
        orthogonalize!(to_measure, site_number)
        to_measure[site_number] *= prime(operator)
        to_measure[site_number] = replaceprime(to_measure[site_number], 2, 1)
        res = real(tr(to_measure))
    elseif length(common_indices) == 2
        to_measure = deepcopy(state)
        site_numbers = get_site_number.(common_indices)
        orthogonalize!(to_measure, minimum(site_numbers))
        local_tensor = to_measure[site_numbers[1]] * to_measure[site_numbers[2]]
        local_tensor_trace = operator * local_tensor
        res = ITensor(1.)
        for site=1:minimum(site_numbers)-1
            res *= tr(to_measure[site])
        end
        res *= local_tensor_trace
        for site=maximum(site_numbers)+1:length(state)
            res *= tr(to_measure[site])
        end
        res = real(scalar(res))
    else
        throw(ArgumentError("can only measure single site and two site operators!"))
    end
    return res
end


"""
    measure_operator(operator::MPO, state::MPS)

Measure an operator for a given state by calculating inner product <state|operator|state>
Operator in MPO language for whole chain
"""
function measure_operator(operator::MPO, state::MPS)
    res = inner(state', operator, state)
    return res
end


"""
    measure_operator(operator::MPO, state::MPO)

Measure an operator for a given state by calculating <operator> = Tr(operator*state),
state is the density matrix.
i)for the MPO-MPO multiplication, prime the indices of the density matrix
ii) trace operators requires prime level of indices to contract: Pair(0,2) (plev = 1)
contracted in MPO-MPO multiplication
"""
function measure_operator(operator::MPO, state::MPO)
    res = tr(operator*prime(state); plev=Pair(0,2))
    return res
end


"""
    apply_single_site_op_to_state!(state::MPS, operator, site_number)

Apply a single-site MPO to an MPS at sites `site`.
"""
function apply_single_site_op_to_state!(state::MPS, operator::ITensor, site_number)
    orthogonalize!(state, site_number)
    state[site_number] = state[site_number] * operator
    noprime!(state[site_number])
    return nothing
end

"""
    apply_single_site_op_to_state!(state::MPO, operator, site_number)

Apply a single-site MPO to an MPO at sites `site``.
"""
function apply_single_site_op_to_state!(state::MPO, operator::ITensor, site_number)
    orthogonalize!(state, site_number)
    # the state has prime levels (0, 1).
    # we first contract the density matrix (0,1) and operator with primed levels (1,2)
    # getting a (0,2) tensor. Afterwards, we apply the h.c. (0,1).
    # result with prime levels (1, 2), which we will reduce by 1 to be back at
    # prime levels (0, 1) as in the beginning
    primed_operator = prime(operator)
    operator_dag = dag(operator)
    state[site_number] = operator_dag*(state[site_number] * primed_operator)
    prime!(state[site_number], -1, "Site")
    return nothing
end


"""
    apply_two_site_op_to_state!(state::MPS, operator, site_number)

Apply a two-site MPO to an MPS at sites `site` and `site + 1`.
kwargs: maxdim and cutoff for svd function
"""
function apply_two_site_op_to_state!(state::MPS, operator::ITensor, site_number::Int64; kwargs...)
    orthogonalize!(state, site_number)
    wf = (state[site_number] * state[site_number+1]) * operator
    noprime!(wf)
    F = safe_svd(wf, inds(state[site_number]); kwargs...)
    state[site_number] = F.U
    state[site_number+1] = F.S * F.V
    debug(logger, "max bond dim after two-site gate: $(maxlinkdim(state))")
    return nothing
end


"""
    apply_two_site_op_to_state!(state::MPO, operator, site_number)

Apply a two-site MPO to an MPS at sites `site` and `site + 1`.
"""
function apply_two_site_op_to_state!(state::MPO, operator::ITensor, site_number::Int64; kwargs...)
    orthogonalize!(state, site_number)
    # the state has prime levels (0, 1).
    # as we apply the operator and its h.c., we need the operator with prime
    # levels (1, 3) and its h.c. with prime levels (0, 2). This will give a
    # result with prime levels (2, 3), which we will reduce by 2 to be back at
    # prime levels (0, 1) as in the beginning
    wf = (state[site_number] * state[site_number+1])
    primed_operator = prime(operator)
    operator_dag = dag(operator)
    wf = operator_dag * (wf * primed_operator)
    prime!(wf, -1, "Site")
    F = safe_svd(wf, inds(state[site_number]); kwargs...)
    state[site_number] = F.U
    state[site_number+1] = F.S * F.V
    debug(logger, "max bond dim after two-site gate: $(maxlinkdim(state))")
    return nothing
end


"""
    get_entanglement_spectrum!(state::Union{MPO, MPS}, bond::Int)

Get the entanglement spectrum of a state for a given bond. The entanglement 
spectrum is the singular values of an SVD of the two-site tensor given by the
matrices adjacent to the bond.
"""
function get_entanglement_spectrum!(state::Union{MPO, MPS}, bond::Int; kwargs...)
    if (bond < 1 || bond > length(state) -1)
        throw(DomainError("invalid bond number $(bond) for state of length $(length(state))"))
    end
    orthogonalize!(state, bond)
    two_site_tensor = state[bond] * state[bond + 1]
    F = safe_svd(two_site_tensor, inds(state[bond]); kwargs...)
    # the contraction leads to higher bond dimension, but this is artificial,
    # make sure to only return as many SVD values as there have been link values
    link_dim = ITensors.dim(commonind(state[bond], state[bond+1]))
    # the entanglement spectrum is the singular values of the given bond
    entanglement_spectrum = zeros(link_dim)
    for i in 1:min(ITensors.dim(F.S, 1), link_dim)
        entanglement_spectrum[i] = F.S[i, i]
    end
    return entanglement_spectrum
end


"""
    get_n_particle_projector(sites, N::Int)

Get the projector onto the N-particle subspace as an MPO.
"""
function get_n_particle_projector(sites, N::Int)
    if N == 0
        zero_state = productMPS(sites, repeat(["Dn"], length(sites)))
        zero_projector = outer(zero_state, zero_state')
        return zero_projector
    end
    states = []
    for comb in collect(combinations(1:length(sites), min(N, length(sites))))
        config = repeat(["Dn"], length(sites))
        config[comb] .= "Up"
        state = productMPS(sites, config)
        push!(states, outer(state, state'))
    end
    return sum(states)
end


"""
    partial_particle_number_measurement(rho::MPO, sites, sub_site_range, N::Int)

Measure the particle number on a subspace defined by `sub_site_range` if the 
total system has a fixed number of N particles. The result is returned as an MPO.
"""
function partial_particle_number_measurement(rho::MPO, sites, sub_site_range, N::Int)
    sub_rho = MPO(rho[sub_site_range])
    rho_ms = []
    for n in 0:min(N, length(sub_site_range))
        proj = get_n_particle_projector(sites[sub_site_range], n)
        proj_sub_rho = apply(apply(proj, sub_rho), proj)
        rho_m = MPO([rho[1:sub_site_range[1]-1]; proj_sub_rho[1:end]])
        push!(rho_ms, rho_m)
    end
    return sum(rho_ms)
end


"""
    sme_increment(rho::MPO, k, dt, dist, X::Itensor, eta, site)

Calculate the increment for an infinitesimal step of the stochastic master 
equation, see Eq. (32) of https://arxiv.org/abs/quant-ph/0611067. 
Note that this version only holds for a measurement of a hermitian operator X!
Moreover, this implementation here assumes that X^X = X.
"""
function sme_increment(rho, k, dt, dist, X, eta, site)
    dW = rand(dist)
    X_exp = measure_operator(X, rho)
    a = sqrt(2 * eta * k) * dW - k * dt
    b = 2 * k * dt
    c = 2 * sqrt(2 * eta * k) * X_exp * dW
    A = left_apply(X, rho[site])
    B = right_apply(rho[site], X)
    C = left_apply(X, B)
    incr = a * (A + B) + b * C - c * rho[site]
    res = copy(rho)
    res[site] = incr
    return res
end