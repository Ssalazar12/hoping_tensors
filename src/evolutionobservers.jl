"""
`EvolutionObserver` are able to observe measurement outcomes during a time 
evolution. Depending on the type of measurements (e.g. operators, entanglement
spectrum at bonds of an MPS/MPO), a different subtype of `EvolutionObserver` is
feasible.
"""
abstract type EvolutionObserver end

"""
An `OperatorObserver` represents an observer which can store measurement outcomes
of measuring operators on an MPO/MPS. For this it knows a number of operators 
which will be measured.
"""
mutable struct OperatorObserver <: EvolutionObserver
    operators::Array
    # TODO: dynamically allocate measurement results in `measurements`? Then we do not need to know
    #  the number of timesteps in advance, it's only used to initialize the storage Array
    number_timesteps::Int
    measurements::Array
    function OperatorObserver(operators::Array, number_timesteps::Int)
        return new(
            operators,
            number_timesteps,
            # the + 1 is to allow a measurement before doing any timesteps
            zeros(number_timesteps + 1, length(operators)),
        )
    end
end

"""
An `EntanglementObserver` observes the entanglement spectrum at all bonds of an 
MPS/MPO and stores it in its field `measurements`. This field is a 3D array with
dimensions (timesteps, bonds, entanglement spectrum values). The max number of 
entanglement spectrum values is limited.
"""
mutable struct EntanglementObserver <: EvolutionObserver
    # TODO: dynamically allocate measurement results in `measurements`? Then we do not need to know
    #  the number of timesteps in advance, it's only used to initialize the storage Array
    number_timesteps::Int
    measurements::Array
    bonds::Array
    function EntanglementObserver(number_timesteps::Int, bonds::Union{Array, UnitRange{Int64}}, max_n_values::Int)
        return new(
            number_timesteps,
            # the + 1 is to allow a measurement before doing any timesteps
            # there are `length(operators) = 1` bonds for the MPS/MPO.
            zeros(number_timesteps + 1, length(bonds), max_n_values),
            bonds
        )
    end
end


"""
A `LinkdimObserver` represents an observer which can store measurement outcomes
of measuring the linkd imensions of an MPS/MPO.
"""
mutable struct LinkdimObserver <: EvolutionObserver
    number_timesteps::Int
    measurements::Array
    function LinkdimObserver(L, number_timesteps::Int)
        return new(
            number_timesteps,
            # the + 1 is to allow a measurement before doing any timesteps
            zeros(number_timesteps + 1, L-1),
        )
    end
end


"""
    measure_by_observer(observer::EvolutionObserver, state, timestep)

Perform a measurement using an `EvolutionObserver`. The observer will measure all its operators, the
results will be stored in the observer's `measurements` storage Array (for each timestep a row).
"""
function measure_by_observer!(observer::EvolutionObserver, state, timestep)
    if timestep + 1 <= size(observer.measurements)[1]
        observer.measurements[timestep+1, :] =
            [measure_operator(op, state) for op in observer.operators]
    else
        # shift all measurements back one timestep, then store the newest
        # measurement at last entry (note that circshift shifts the first
        # dimension by default)
        observer.measurements = circshift(observer.measurements, -1)
        observer.measurements[end, :] =
            [measure_operator(op, state) for op in observer.operators]
    end
    return nothing
end

"""
    measure_by_observer(observer::EntanglementObserver, state, timestep)

Perform a measurement using an `EntanglementObserver`. The observer measures the
entanglement spectrum at every bond of the MPS/MPO.
"""
function measure_by_observer!(observer::EntanglementObserver, state, timestep)
    if timestep + 1 <= size(observer.measurements)[1]
        for (i, bond) in enumerate(observer.bonds)
            es = get_entanglement_spectrum!(state, bond)
            observer.measurements[timestep+1, i, 1:min(length(es), size(observer.measurements)[3])] = 
                es[1:min(length(es), size(observer.measurements)[3])]
        end
    else
        # shift all measurements back one timestep, then store the newest
        # measurement at last entry (note that circshift shifts the first
        # dimension by default)
        observer.measurements = circshift(observer.measurements, -1)
        for (i, bond) in enumerate(observer.bonds)
            es = get_entanglement_spectrum!(state, bond)
            observer.measurements[end, i, 1:min(length(es), size(observer.measurements)[3])] = 
                es[1:min(length(es), size(observer.measurements)[3])]
        end
    end
    return nothing
end


"""
    measure_by_observer(observer::LinkdimObserver, state, timestep)

Perform a measurement using an `LinkdimObserver`. The observer will measure the
link dimensions at all bonds.
"""
function measure_by_observer!(observer::LinkdimObserver, state, timestep)
    if timestep + 1 <= size(observer.measurements)[1]
        observer.measurements[timestep+1, :] = linkdims(state)
    else
        observer.measurements = circshift(observer.measurements, -1)
        observer.measurements[end, :] = linkdims(state)
    end
    return nothing
end