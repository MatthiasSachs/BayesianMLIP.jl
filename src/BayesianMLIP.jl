module BayesianMLIP 
    include("./miniACEflux/miniACEflux.jl")
    include("./nlmodels.jl")
    include("./MHoutputschedulers.jl")
    include("./outputschedulers.jl")
    include("./utils.jl")
    include("./dynamics.jl")
    include("./sampler.jl")
    include("./json_parser.jl")

end