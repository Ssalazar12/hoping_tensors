# we dispatch LinearAlgebra's diag function, so explicitly import it
import LinearAlgebra.diag

logger = getlogger(@__MODULE__)

"""
    get_site_number(index::Index)

Helper function to get the site number of a site index. This would actually be nicer if part of
ITensors (should we make a pull request or is this already implemented somewhere?).
"""
function get_site_number(index::Index)
    tags = index.tags
    site_number = nothing
    for i in 1:length(tags)
        # Site number tag is "n=i"
        if occursin("n=", String(tags[i]))
            site_number = parse(Int, split(String(tags[i]), "=")[end])
        end
    end
    if isnothing(site_number)
        throw(ArgumentError(index, "Index does not have a site number"))
    else
        return site_number
    end
end

"""
    diag(mpo::MPO)

Get the diagonal of an MPO. The resulting MPO has the same structure, dimensions and
legs. It is expected that the site-tensors of the MPO have exactly two physical indices.
The resulting MPO has zero aalues wheneaer the physical indices are not equal, else the
entries of the input MPO.
"""
function diag(mpo::MPO)
    diag_mpo = deepcopy(mpo)
    for (site_number, site_tensor) in enumerate(diag_mpo)
        diag_mpo[site_number] = diag(site_tensor)
    end
    return diag_mpo
end

"""
    diag(site_tensor::ITensor)

Get the diagonal of an ITensor with exactly two physical legs and one or two bond legs.
The diagonal ITensor will haae the same indices, but aalues are zero wheneaer the two
physical indices are not equal.
"""
function diag(site_tensor::ITensor)
    diag_tensor = deepcopy(site_tensor)
    indices = inds(diag_tensor)
    site_inds = [i for i in indices if hastags(i, "Site")]
    link_inds = [i for i in indices if hastags(i, "Link")]

    length(site_inds) == 2 ||
        throw(DimensionMismatch("need two site indices in an MPO site-tensor"))
    length(link_inds) in [1 2] ||
        throw(DimensionMismatch("need one or two link indices in an MPO site-tensor"))


    # make sure that indices are in correct order before setting slices
    diag_tensor = permute(diag_tensor, site_inds..., link_inds...)

    for site_one in 1:space(site_inds[1])
        for site_two in 1:space(site_inds[2])
            if site_one == site_two
                # diagonal entry, so do nothing
                continue
            else
                # note that eaen if diag_tensor has 3 legs, this works
                diag_tensor[site_one, site_two, :, :] =
                    zeros(space.([i for i in link_inds])...)
            end
        end
    end
    return diag_tensor
end


"""
    order_array!(a::Array)

Order columns of a 2D Array `a` by continuosity of the values.
"""
function order_array!(a::Array)
    length(size(a)) == 2 || throw(DimensionMismatch("Can only order columns of 2D array!"))
    for i=1:size(a, 1) - 1
        a[i+1, :] = order_by_closest(a, i)
    end
    for i=1:size(a, 1) - 2
        a[i+2, :] = order_by_continuity(a, i)
    end
    return nothing
end


"""
    order_by_closest(a::Array, i)

Order column `i+1` of Array `a` by closeness of values to column `i`. This 
means that in the resulting array b, the entry b[i+1, j] is the one closest to
the value a[i, j].
"""
function order_by_closest(a::Array, i)
    col = a[i, :]
    next_col = a[i+1, :]
    ordered_next_col = deepcopy(next_col)
    for j=1:length(ordered_next_col)
        entry = col[j]
        dists = abs.(ordered_next_col[j:end] .- entry)
        min_index = argmin(dists) + (j - 1)
        if min_index != j
            ordered_next_col[j], ordered_next_col[min_index] = ordered_next_col[min_index], ordered_next_col[j]
        end
    end
    return ordered_next_col
end


"""
    order_by_continuity(a::Array, i)

Order column `a[i+2, :]` by continuity. This means that we order the column such
that the resulting entry `b[i+2, j]` is chosen such that it's the most 
continuous continuation of the entries `a[i+1, j]` and `a[i, j]`
"""
function order_by_continuity(a::Array, i)
    ordered_col = deepcopy(a[i+2, :])
    for j=1:length(ordered_col)
        previous_entry = a[i+1, j]
        previous_dy = a[i+1, j] - a[i, j]
        dists = abs.((ordered_col[j:end] .- previous_entry) .- previous_dy)
        min_index = argmin(dists) + (j - 1)
        if min_index != j
            ordered_col[j], ordered_col[min_index] = ordered_col[min_index], ordered_col[j]
        end
    end
    return ordered_col
end


"""
    safe_svd(tensor::ITensor, indices; kwargs...)

SVD routine with fallback onto ITensor custom SVD if no convergence is achieved
with standard SVD.
"""
function safe_svd(tensor::ITensor, indices; kwargs...)
    F = svd(tensor, indices; kwargs...)
    if isnothing(F)
        F = svd(tensor, indices; kwargs..., alg="recursive")
    end
    return F
end


"""
    get_reduced_mpo(state::MPO, sites, bond::Int, link_index_val::Int; subsystem="right")

Returns the MPO of the reduced state of either the left or right subsystem w.r.t.
a given bond. The MPO is obtained by an SVD at the given bond. Then only the 
reduced tensor to the left/right of the bond corresponding to a given link index 
value is kept.
"""
function get_reduced_mpo(state::MPO, sites, bond::Int, link_index_val::Int; subsystem="right")
    F = safe_svd(state[bond] * state[bond+1], inds(state[bond]))
    if subsystem == "right"
        new_site_tensor = F.V
        subsystem_range = bond+1:length(sites)
        site_indices = commoninds(new_site_tensor, state[bond+1])
        link_ind = commonind(new_site_tensor, F.S)
    else
        new_site_tensor = F.U
        subsystem_range = 1:bond
        site_indices = commoninds(new_site_tensor, state[bond])
        link_ind = commonind(new_site_tensor, F.S)
    end
    # now loop over the new site tensor given by the SVD (U or V), keeping
    # the link index fixed
    new_site_matrix = zeros(ITensors.dim.(site_indices)...)
    site_index_ranges = [1:i for i in ITensors.dim.(site_indices)]
    for index_tuple in Base.Iterators.product(site_index_ranges...)
        index_value_paris = site_indices .=> index_tuple
        new_site_matrix[index_tuple...] = new_site_tensor[index_value_paris..., link_ind => link_index_val]
    end
    # get the new subsystem tensor by contracting the new site tensor with all
    # existing site tensors of the rest of the subsystem
    reduced_tensor = ITensor(new_site_matrix, site_indices)
    for i in subsystem_range[2:end]
        reduced_tensor *= state[i]
    end
    reduced_mpo = MPO(reduced_tensor, sites[subsystem_range])
    return reduced_mpo
end


"""
    number_of_inds(A::ITensor)

Get the number of unique indices of a n ITensor (not counting prime levels).
"""
function number_of_inds(A::ITensor)
    return length(unique(id.(inds(A))))
end


"""
    pure(rho::MPO, n_particles, sites)

Get a pure MPO from a given (mixed) one. We get the pure state with same local
densities as `pure = sum_i sqrt(rho[i,i]) |i>`.
"""
function pure(rho::MPO, n_particles, sites; tol=1e-10)
    configs = collect(combinations(1:length(rho), n_particles))
    n_diag_entries = length(configs)
    diag_entries = zeros(n_diag_entries)
    basis_states = []
    for (i, config) in enumerate(configs)
        spin_config = fill("Dn", length(rho))
        spin_config[config] .= "Up"
        state = productMPS(sites, spin_config)
        push!(basis_states, state)
        diag_entry = real(inner(state', rho, state))
        if abs(diag_entry) < tol
            diag_entry = 0
        end
        diag_entries[i] = sqrt(diag_entry)
    end
    if all(diag_entries .== 0)
        throw(ArgumentError("no overlap of the given state with pure states of $n_particles particles"))
    end
    pure_state = sum(diag_entries .* basis_states)
    pure_rho = outer(pure_state, pure_state')
    debug(logger, "max bond dim of state: $(maxlinkdim(rho)); max bond dim of pure state: $(maxlinkdim(pure_rho))")
    return pure_rho
end


"""
    get_indices_of_degenerate_values(a; tol=1e-6)

Get the indices of the degenerate values of a vector `a`.
"""
function get_indices_of_degenerate_values(a; tol=1e-6)
    dist_matrix = collect([abs.(circshift(a, shift) - a) for shift in 1:length(a) - 1])
    inds = any(permutedims(hcat(dist_matrix...)) .< tol, dims=1)
    return inds[1, :]
end


"""
    get_sum_of_degenerate_values(a; dim=2)

Get the sum of degenerate values of a 2D array `a` along a given dimension.
"""
function get_sum_of_degenerate_values(a; dims=2, tol=1e-6)
    if dims == 2
        len = size(a)[1]
    else
        len = size(a)[2]
    end
    res = zeros(len)
    for i in 1:len
        slice = (dims == 2 ? a[i, :] : a[:, i])
        inds = get_indices_of_degenerate_values(slice; tol=tol)
        res[i] = sum(dims == 2 ? a[i, inds] : a[inds, i])
    end
    return res
end


"""
    left_apply(A::ITensor, B::ITensor)

Calculates s'-A-B-s = s'-A-s * s'-B-S by keeping direction of B invariant.
This guarantees that operations like X * rho preserve the direction and prime
levels of rho.
"""
function left_apply(A::ITensor, B::ITensor)
    a = replaceprime(A, 1 => 2)
    res = a * B
    res = replaceprime(res, 2 => 0)
    return res
end

"""
    right_apply(A::ITensor, B::ITensor)

Calculates s'-A-B-s = s'-A-s * s'-B-S by keeping direction of A invariant.
This guarantees that operations like rho * X preserve the direction and prime
levels of rho.
"""
function right_apply(A::ITensor, B::ITensor)
    b = replaceprime(B, 0 => 2)
    res = A * b
    res = replaceprime(res, 2 => 1)
    return res
end