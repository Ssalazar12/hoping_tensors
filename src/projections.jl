logger = getlogger(@__MODULE__)

"""
    function project(state::MPS, rho::MPO, site_range::UnitRange{Int})

Get a partial density matrix by projection onto a subspace. This function does
the following contraction.

1. If `site_range` is at the beginning of the sites of `rho`:

state[1]--...--state[end]
   |              |          |
 rho[1]--....---rho[i]---rho[i+1]--...
   |              |          |
state[1]'-...--state[end]'

Here, `i` == site_range[end]. The contraction is done from the left to the right.

2. If `site_range` is at the end of the sites of `rho`:

               state[1]--...--state[end]
       |          |              |
...--rho[i-1]---rho[i]--....---rho[end]
       |          |              |
               state[1]'-...--state[end]'

Here, `i` == site_range[1]. The contraction is done from the right to the left.
"""
function project(state::MPS, rho::MPO, site_range::UnitRange{Int})
    if site_range[1] == 1
        return project_at_start(state, rho, site_range)
    elseif site_range[end] == length(rho)
        return project_at_end(state, rho, site_range)
    else
        throw(ArgumentError( "site_range should be at beginning or end of the MPO"))
    end
end


"""
    function project_at_start(state::MPS, rho::MPO, site_range::UnitRange{Int})

Get a partial density matrix by projection onto a subspace. This function does
the following contraction:

state[1]--...--state[end]
   |              |          |
 rho[1]--....---rho[i]---rho[i+1]--...
   |              |          |
state[1]'-...--state[end]'

Here, `i` == site_range[end]. The contraction is done from the left to the right.
"""
function project_at_start(state::MPS, rho::MPO, site_range::UnitRange{Int})
    L = ITensor(1.)
    for site in site_range
        L *= state[site]
        L *= rho[site]
        L *= prime(state[site])
    end
    sub_rho = MPO(rho[site_range[end] + 1:end])
    sub_rho[1] *= L
    return sub_rho
end


"""
    function project_at_end(state::MPS, rho::MPO, site_range::UnitRange{Int})

Get a partial density matrix by projection onto a subspace. This function does
the following contraction:

               state[1]--...--state[end]
       |          |              |
...--rho[i-1]---rho[i]--....---rho[end]
       |          |              |
               state[1]'-...--state[end]'

Here, `i` == site_range[1]. The contraction is done from the right to the left.
"""
function project_at_end(state::MPS, rho::MPO, site_range::UnitRange{Int})
    L = ITensor(1.)
    for site in 0:length(site_range)-1
        L *= state[end - site]
        L *= rho[end - site]
        L *= prime(state[end - site])
    end
    sub_rho = MPO(rho[1:site_range[1] - 1])
    sub_rho[end] *= L
    return sub_rho
end