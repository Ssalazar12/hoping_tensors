logger = getlogger(@__MODULE__)

"""
An `EvolutionSweep` represents a single sweep through a physical chain. The 
sweep has a vector of site_numbers which determines the site order in evolution. 
In a sweep, its `gates` will be a pplied to the state.
"""
struct EvolutionSweep
    gates::Vector
    site_numbers::Vector
    maxdim::Int
    cutoff::Float64
end


struct InefficientMeasurementSweep
    ops::Vector
    site_numbers::Vector
    dist::Normal{Float64}
    k::Float64
    dt::Float64
    maxdim::Int
    cutoff::Float64
    eta::Float64
end

"""
    do_evolution_sweep!(state, sweep::EvolutionSweep)

Perform a sweep on a state by applying the sweep's gates to the defined site_numbers.
"""
function do_evolution_sweep!(state, sweep::Union{EvolutionSweep, InefficientMeasurementSweep})
    if isa(sweep, EvolutionSweep)
        for (i, gate) in enumerate(sweep.gates)
            site_number = sweep.site_numbers[i]
            if isa(gate, Array)
                # the gate is a collection of Kraus ops that have to be applied
                # simultaneously
                apply_kraus_operators_to_state!(
                    state,
                    gate,
                    site_number
                )
            else
                apply_gate_to_state!(
                    state,
                    gate,
                    site_number;
                    maxdim=sweep.maxdim,      
                    cutoff=sweep.cutoff,
                )
            end
        end
    else
        for (i, op) in enumerate(sweep.ops)
            incr = sme_increment(state, sweep.k, sweep.dt, sweep.dist, 
                op, sweep.eta, sweep.site_numbers[i])
            res = state + incr
            for i=eachindex(res)
                # have to sweep through to change the input `state`, to not have
                # to return it (because the above implementation does not rely
                # on any return)
                state[i] = res[i]
            end
        end
    end
    return nothing
end

"""
    create_st1_tebd_sweeps(two_site_ampo, sites, dt, maxdim, cutoff)

Create the two sweeps necessary for a first order Suzuki-Trotter TEBD algorithm. 
The time evolution operator is decomposed into unitary two-site evolution 
operators. In a first sweep, these operators are applied to the odd sites, in a 
second sweep, to the even sites. The resulting error for a single timestep dt is 
then O(dt).
"""
function create_st1_tebd_sweeps(two_site_ampo, sites, dt, maxdim, cutoff)
    gates = create_gates_from_two_site_ampo(two_site_ampo, sites, dt)
    odd_site_numbers = 1:2:length(sites)-1
    odd_sweep = EvolutionSweep(gates[odd_site_numbers], odd_site_numbers, maxdim, cutoff)
    # sweep over even sites should be done from right to left
    even_site_numbers = reverse(2:2:length(sites)-1)
    even_sweep = EvolutionSweep(gates[even_site_numbers], even_site_numbers, maxdim, cutoff)
    return [odd_sweep, even_sweep]
end

"""
    create_st2_tebd_with_dissipators_sweeps(two_site_ampo, sites, dt, maxdim, cutoff)

Create sweeps necessary for the time evolution as in PRR 2, 043052, eq.(16)
"""
function create_st2_tebd_with_dissipators_sweeps(single_site_ampo_array::Vector,
    two_site_ampo_array::Vector, 
    sites, 
    impurity_sites_number::Vector,
    site_numbers,
    dt::Number, 
    γ, 
    maxdim::Int64,
    cutoff::Number;
    kwargs...)
# create dissipative evolution sweeps
    dissipative_gates, dissipative_gates_site_number = create_dissipative_kraus_p_h_gates_fermions(sites,
                                                            impurity_sites_number, dt, γ; kwargs...)
    #create sweeps corresponding to left part of Linbladian time ev as in PRB
    dissipative_sweep_left = EvolutionSweep(dissipative_gates, dissipative_gates_site_number , maxdim, cutoff)
    

    dissipative_sweep_right =  EvolutionSweep(reverse(dissipative_gates), reverse(dissipative_gates_site_number), maxdim, cutoff)
    #create Trotter-Suzuki Hamiltonian step
    hamiltonian_single_gates = create_gates_from_single_site_ampo(single_site_ampo_array, sites, dt)
    single_site_sweep = EvolutionSweep(hamiltonian_single_gates, site_numbers,
    maxdim, cutoff)

    hamiltonian_gates = create_gates_from_two_site_ampo(two_site_ampo_array, sites, dt)
    odd_site_numbers = 1:2:length(sites)-1
    even_site_numbers = 2:2:length(sites)-1
    odd_sweep = EvolutionSweep(hamiltonian_gates[odd_site_numbers], odd_site_numbers,maxdim, cutoff)
    even_sweep = EvolutionSweep(hamiltonian_gates[even_site_numbers], even_site_numbers, maxdim, cutoff)
    #join all sweeps: first left dissipative, than odd and even Hamiltonian, than rigt dissipative
    all_sweeps = vcat(vcat(dissipative_sweep_left, [single_site_sweep, odd_sweep, even_sweep]),
    dissipative_sweep_right)
    return all_sweeps
end
