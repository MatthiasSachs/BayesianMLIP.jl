using ACE, ACEatoms, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2, JuLIP, Plots, StaticArrays, Distributions
import StatsBase: sample
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics  
using BayesianMLIP.MiniACEflux, BayesianMLIP.Utils, BayesianMLIP.Samplers, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser
using Random: seed!, rand
import BayesianMLIP.NLModels: Hamiltonian, energy, forces
# using BayesianMLIP.MiniACEflux: FluxPotential
using ACEflux: FluxPotential
import Distributions: logpdf, MvNormal
using JSON

# Initialize Finnis-Sinclair Model with ACE basis (w/ coefficients=0)
FS(ϕ) = ϕ[1] # + sqrt(abs(ϕ[2]) + 1/9) - 1/3
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 1), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 3.0);      

# Initialize atomic configuration
at = bulk(:Cu, cubic=true) * 3;
rattle!(at, 0.1) ;       # If we rattle this too much, the integrators can become unstable

BAOAB_Sampler = BAOAB(0.01, pot, at; γ=1.0, β=1.0)

function generate_data(Ndata::Int64, sampler, _at::AbstractAtoms, filename::String) 
    # Initialize pot with random parameters 
    _FS(ϕ) = ϕ[1] # + sqrt(abs(ϕ[2]) + 1/9) - 1/3
    _model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 1), GenLayer(_FS), sum);
    _pot = ACEflux.FluxPotential(_model, 3.0);
    basis = Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 1).m.basis
    scaling = ACE.scaling(basis, 2)
    scaling[1] = 1.
    θ = randn(nparams(_pot)) ./ scaling
    BayesianMLIP.NLModels.set_params!(_pot, θ)

    data = [] 
    ωE = 1.0
    ωF = ωE/(3*length(_at))
    for i in 1:Ndata 
        Dynamics.run!(sampler, _pot, _at, 1000; outp=nothing) 
        # push data (without Gaussian noise for now) 
        Energy = energy(_pot, _at) 
        Noisy_Energy = Energy + sqrt(1/ωE) * randn()
        Forces = forces(_pot, _at) 
        Noisy_Forces = Forces + sqrt(1/ωF) * [randn(3) for _ in 1:length(_at)]
        push!(data, (at=deepcopy(_at), E = Noisy_Energy, F = Noisy_Forces))
        
        println("Data added: ", i)
    end 

    Theta = BayesianMLIP.NLModels.get_params(_pot) 
    Data = data 

    dict = Dict{String, Any}("theta" => Theta, "data"  => Data)
    save("./Run_Data/Artificial_Data/" * filename * ".jld2", dict)
    return (Theta, Data) 
end 

generate_data(30, BAOAB_Sampler, at, "artificial_data_Noisy_30") ;

