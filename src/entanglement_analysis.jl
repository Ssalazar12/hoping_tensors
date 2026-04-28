logger = getlogger(@__MODULE__)

"""
    get_indices_of_coherence_values_by_mixed_limit(state; n_interpolation_steps=100, tol=1e-6)

For a state, get the indices of those entanglement spectrum values that vanish
for full decoherence.
"""
function get_indices_of_coherence_values_by_mixed_limit(state, bonds; n_interpolation_steps=100, tol=1e-6)
    diag_state = diag(state)
    es_observer = interpolate_and_measure_entanglement(
        state, diag_state, bonds; n_interpolation_steps=n_interpolation_steps
    )
    # es_values is a 3D array with dims (timestep, length(bonds), max_bond_dim)
    es_values = es_observer.measurements
    indices = zeros(Bool, size(es_values, 2), size(es_values, 3))
    for i=1:size(es_values, 2)
        order_array!(es_values[:, i, :])
        # we're interested in those spectral values that are non-zero for the
        # state but vanish for the fully decohered state
        indices[i, :] = (es_values[end, i, :] .< tol) .& (es_values[1, i, :] .> 0)
    end
    return indices
end

"""
    get_indices_of_coherence_values_by_pure_limit(state, n_particles, sites, bonds; n_interpolation_steps=100, tol=1e-6)

For a state, get the indices of those entanglement spectrum values that render
degenerate for the pure limit.
"""
function get_indices_of_coherence_values_by_pure_limit(state, n_particles, sites, bonds; n_interpolation_steps=100, tol=1e-6)
    pure_state = pure(state, n_particles, sites)
    es_observer = interpolate_and_measure_entanglement(
        state, pure_state, bonds; n_interpolation_steps=n_interpolation_steps
    )
    # es_values is a 3D array with dims (timestep, bond, max_bond_dim)
    es_values = es_observer.measurements
    indices = zeros(Bool, size(es_values, 2), size(es_values, 3))
    for i=1:size(es_values, 2)
        order_array!(es_values[:, i, :])
        # we're interested in those spectral values that are non-zero for the
        # state but vanish for the fully decohered state
        degenerate_with_next = [abs.(es_values[end, i, 1:end-1] .- es_values[end, i, 2:end]) .< tol; false]
        degenerate_with_prev = [false; abs.(es_values[end, i, 2:end] .- es_values[end, i, 1:end-1]) .< tol]
        indices[i, 1:end] = (degenerate_with_next .| degenerate_with_prev)
    end
    return indices
end


"""
    interpolate_and_measure_entanglement(state_1::Union{MPS, MPO}, state_2::Union{MPS, MPO}, bonds; n_interpolation_steps=100)

Implements 
 (1-p) * state_1 + p * state_2
for p in [0, 1] and measures the entanglement spectrum throughout the interpolation.
"""
function interpolate_and_measure_entanglement(state_1::Union{MPS, MPO}, state_2::Union{MPS, MPO}, bonds; n_interpolation_steps=100)
    if (typeof(state_1) != typeof(state_2)) || (length(state_1) != length(state_2))
        throw(TypeError("Both states have to be of the same type and same length for interpolation!"))
    end

    max_bond_dim = maximum(vcat([maxdim(tensor) for tensor in state_1], 
                                [maxdim(tensor) for tensor in state_2]))
    es_observer = EntanglementObserver(n_interpolation_steps, bonds, max_bond_dim)
    interpolation_range = LinRange(0, 1, n_interpolation_steps + 1)
    for (step, p) in enumerate(interpolation_range)
        if p == 0
            interpolated_state = state_1
        elseif p == 1
            interpolated_state = state_2
        else
            interpolated_state = (1 - p) * state_1 + p * state_2
        end
        # note that the observer starts counting steps at 0
        measure_by_observer!(es_observer, interpolated_state, step - 1)
        debug(logger, "max bond dim of interpolated state for p=$p: $(maxlinkdim(interpolated_state))")
    end
    return es_observer
end


"""
    get_entanglement_by_interpolation!(state::MPO, bond::Int)

Obtain single-valued entanglement measure of `state` for a bipartition at `bond`.
"""
function get_entanglement_by_interpolation!(state::MPO, bond::Int)
    inds = get_indices_of_coherence_values_by_mixed_limit(state, [bond])
    spectrum = get_entanglement_spectrum!(state, bond)
    # for the measure we need the sum over the spectrum squared
    entanglement = sum(spectrum[inds[1, 1:length(spectrum)]].^2)
    return entanglement
end


"""
    function get_oses_value_of_left_subsystem_empty(sites::Vector{Index{Int64}}, state::MPO, bond::Int)

Get the OSES value corresponding to the left subsystem emtpy. This value can be 
obtained by projecting onto the left-side-empty subspace and then take the 
purity (i.e. norm squared) of the remaining partial tensor.
"""
function get_oses_value_of_left_subsystem_empty(sites::Vector{Index{Int64}}, state::MPO, bond::Int)
    psi_empty = productMPS(sites[1:bond], repeat(["Dn"], bond))
    sub_state = project(psi_empty, state, 1:bond)
    return norm(sub_state)^2
end

"""
    function get_oses_value_of_right_subsystem_empty(sites::Vector{Index{Int64}}, state::MPO, bond::Int)

Get the OSES value corresponding to the right subsystem emtpy. This value can be 
obtained by projecting onto the right-side-empty subspace and then take the 
purity (i.e. norm squared) of the remaining partial tensor.
"""
function get_oses_value_of_right_subsystem_empty(sites::Vector{Index{Int64}}, state::MPO, bond::Int)
    psi_empty = productMPS(sites[bond+1:end], repeat(["Dn"], length(sites)-bond))
    sub_state = project(psi_empty, state, bond+1:length(sites))
    return norm(sub_state)^2
end


"""
    get_configuration_coherence(rho::MPO, sites, sub_site_range, N::Int)

Get the configuration coherence as the relative purity of the inital state and 
a local particle number measurement. This is only defined for a state rho with 
a fixed number of N particles.
"""
function get_configuration_coherence(rho::MPO, sites, sub_site_range, N::Int)
    purity_rho = tr(apply(rho, rho))
    rho_m = partial_particle_number_measurement(rho, sites, sub_site_range, N)
    purity_rho_m = tr(apply(rho_m, rho_m))
    return real(purity_rho - purity_rho_m)
end


"""
    partial_transpose(A::MPO, sites)

Get the partial transpose of an MPO by swapping indices for a subset of sites.
"""
function partial_transpose(A::MPO, sites)
    A = copy(A)
    for n in sites
      A[n] = swapinds(A[n], siteinds(A, n)...)
    end
    return A
end


"""
    negativity(A::MPO, sites)

Calculate the negativity of an MPDO by diagonalizing the partial transpose.
Note that this is a very numerically heavy calculation, only use for small 
system sizes.
"""
function negativity(A::MPO, sites)
    pt = partial_transpose(A, sites)
    t = pt[1]
    for i=2:length(pt)
        t *= pt[i]
    end
    N = length(A)
    mat = reshape(Array(t, siteinds(A, plev=0), siteinds(A, plev=1)), 2^N, 2^N)
    ev = eigvals(mat)
    return abs(sum(abs.(ev)) - sum(ev)) / 2
end