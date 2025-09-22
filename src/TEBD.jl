module TEBD

using ITensors, ITensorMPS
using Combinatorics
using Memento
using Distributions
using LinearAlgebra

const LOGGER = getlogger(@__MODULE__)

function __init__()
    # setup logging
    Memento.register(LOGGER)
    try
        # we use try/catch because this raises an error if the JULIA_DEBUG env 
        # var is not defined
        if parse(Bool, ENV["JULIA_DEBUG"])
            setlevel!(LOGGER, "debug")
        else
            setlevel!(LOGGER, "warn")
        end
    catch
        setlevel!(LOGGER, "warn")
    end
end

include("exports.jl")

include("dissipators.jl")
include("entanglement_analysis.jl")
include("evolution.jl")
include("evolutionobservers.jl")
include("evolutionsweeps.jl")
include("gates.jl")
include("measurements.jl")
include("projections.jl")
include("utilities.jl")

end # module
