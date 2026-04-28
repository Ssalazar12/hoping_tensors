"""
    do_timestep!(state, evolution_sweeps::Vector, observers::Vector, timestep::Int)

Do a single timestep consisting of any number of sweeps. After the timestep is done with its sweeps,
measurements are performed on the resulting state. Every observer will perform its own measurement.
"""
function do_timestep!(state, evolution_sweeps::Vector, observers::Vector, timestep::Int)
    # TODO: when measurement results can be dynamically stored by observer, we do not need to
    #  explicitly pass `timestep`
    for sweep in evolution_sweeps
        do_evolution_sweep!(state, sweep)
    end
    debug(logger, "max bond dim after timestep $timestep: $(maxlinkdim(state))")
    for observer in observers
        measure_by_observer!(observer, state, timestep)
    end
end
