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
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
model = Chain(Linear_ACE(;ord = 1, maxdeg = 1, Nprop = 2), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 6.0);
# Don't need to initialize this since it'll be assigned in run, but for testing purposes 
set_params!(pot, randn(nparams(pot)))         

# Initialize atomic configuration
at = bulk(:Cu, cubic=true) * 3;
rattle!(at, 0.1) ;       # If we rattle this too much, the integrators can become unstable

# Testing parameters (NLModels), energy/forces (JuLIP) functions 
get_params(pot)
nparams(pot)
energy(pot, at)
forces(pot, at)

# Run BAOAB to sample from Gibbs measure of Langevin dynamics with ACE potential
BAOAB_Sampler = BAOAB(0.01, pot, at; γ=1.0, β=1.0)
BAOsteps = []
Dynamics.run!(BAOAB_Sampler, pot, at, 500; outp = BAOsteps) 
x_traj1 = [step.X[1][1] for step in BAOsteps]
plot(1:length(BAOsteps), x_traj1, title="BAOAB Component Trajectory", legend=false)
histogram(x_traj1, bins = :scott, title="Histogram of BAOAB Samples", legend=false)
plot(1:length(x_traj1), [Hamiltonian(pot, elem) for elem in BAOsteps], title="BAOAB Hamiltonian", legend=false)

# Run BADODADB 
BADODAB_Sampler = BADODAB(0.1, pot, at)
BADODABsteps = []
Dynamics.run!(BADODAB_Sampler, pot, at, 500; outp = BADODABsteps)
x_traj2 = [step.X[1][2] for step in BADODABsteps]
plot(1:length(BADODABsteps), x_traj2, title="BADODAB Component Trajectory", legend=false)
histogram(x_traj2, bins = :scott, title="Histogram of BADODAB Samples", legend=false)
plot(1:length(BADODABsteps), [Hamiltonian(pot, elem) for elem in BADODABsteps], title="BADODAB Hamiltonian", legend=false)



function generate_data(Ndata::Int64, sampler, _at::AbstractAtoms, filename::String) 
    # Initialize pot with random parameters 
    _FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
    _model = Chain(Linear_ACE(;ord = 1, maxdeg = 1, Nprop = 2), GenLayer(_FS), sum);
    _pot = ACEflux.FluxPotential(_model, 6.0);
    θ = randn(nparams(_pot))
    set_params!(_pot, θ)

    data = [] 
    for i in 1:Ndata 
        Dynamics.run!(sampler, _pot, _at, 1000; outp=nothing) 
        # push data (without Gaussian noise for now) 
        Energy = energy(_pot, _at) 
        Forces = forces(_pot, _at) 
        push!(data, (at=deepcopy(_at), E = Energy, F = Forces))
        println("Data added: ", i)
    end 

    Theta = get_params(_pot) 
    Data = data 

    dict = Dict{String, Any}("theta" => Theta, "data"  => Data)
    save("./Run_Data/Artificial_Data/" * filename * ".jld2", dict)
    return (Theta, Data) 
end 

# generate_data(5, BAOAB_Sampler, at, "artificial_data2")

