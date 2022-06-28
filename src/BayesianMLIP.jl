module BayesianMLIP 
    include("./miniACEflux/miniACEflux.jl")
    include("./nlmodels.jl")
    include("./MHoutputschedulers.jl")
    include("./outputschedulers.jl")
    include("./dynamics.jl")
    include("./utils.jl")
    include("./sampler.jl")

end