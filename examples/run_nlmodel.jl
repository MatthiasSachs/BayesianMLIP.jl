using ACE, ACEatoms, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2, JuLIP, Plots, StaticArrays, StatsBase
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics         
using BayesianMLIP.Utils, BayesianMLIP.MiniACEflux
using Random: seed!, rand
using ACE: O3, evaluate, val, SymmetricBasis, State
using BayesianMLIP.NLModels: get_params!, set_params!, Energy, Forces, gradParams, Hamiltonian
using BayesianMLIP.MiniACEflux: FluxPotential
using Distributions
using Distributions: logpdf, MvNormal, Normal 


# Initialize Finnis-Sinclair Model with ACE basis (w/ coefficients=0)
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 2), GenLayer(FS), sum);
pot = FluxPotential(model, 6.0);
θ = randn(30)
set_params!(pot, θ)

# Initialize atomic configuration
at = bulk(:Cu, cubic=true) * 3 ;
rattle!(at,0.6) ;
at2 = deepcopy(at) ;

# Testing Energy, Forces, and gradParams Functions
Energy(pot, at)
Forces(pot, at)
gradParams(pot, at, θ)

# Run BAOAB to sample from Gibbs measure of Langevin dynamics with ACE potential
BAOAB_Sampler = BAOAB(0.1, pot, at; γ=1.0, β=1.0)
BAOsteps = []
run!(BAOAB_Sampler, pot, at, 1000; outp = BAOsteps) 

# Check trajectory of particle position component
x_traj = [step.X[1][1] for step in BAOsteps]
plot(1:length(x_traj), x_traj, title="BAOAB Component Trajectory", legend=false)
histogram(x_traj, bins = :scott, title="Histogram of BAOAB Samples", legend=false)

# Check if Hamiltonian is approximately conserved in oscillation
# Usually starts off at high Hamiltonian and stabilizes after few dozen steps
plot(1:length(x_traj), [Hamiltonian(pot, elem) for elem in BAOsteps], title="BAOAB Hamiltonian")

# Check behavior of energy: should be converging at local minima and moving to other local minima
plot(1:length(x_traj), [Energy(pot, elem) for elem in BAOsteps], title="BAOAB Energy")




# Run BADODADB 
BADO_Sampler = BADODAB(0.1, pot, at2)
BADOsteps = []
run!(BADO_Sampler, pot, at2, 1000; outp = BADOsteps)

# Check trajectory of particle position component
x_traj2 = [step.X[1][1] for step in BADOsteps]
plot(1:length(x_traj2), x_traj2, title="BADODAB Component Trajectory")
histogram(x_traj2, bins = :scott, title="Histogram of BADODAB Samples")

# Check if Hamiltonian is approximately conserved in oscillation
# Usually starts off at high Hamiltonian and stabilizes after few dozen steps
plot(1:length(x_traj2), [Hamiltonian(pot, elem) for elem in BADOsteps], title="BADODAB Hamiltonian")

# Check behavior of energy: should be converging at local minima and moving to other local minima
plot(1:length(x_traj2), [Energy(pot, elem) for elem in BADOsteps], title="BADODAB Energy")


using BayesianMLIP.Utils: StatisticalModel, params, get_gll, get_glp

function generate_data(Ndata::Int64, sampler, model, at) 
    energy_std = 2.687172212269082      # 0.05 times the mean of 280 energies in dataset1
    forces_std = 0.012395149775211672   # 0.05 times the mean of the norm of 280*32*3 components of each energy in dataset1
    data = []   # data = Dict("theta" => theta, "data" => data)
    for k = 1:Ndata
        BayesianMLIP.Dynamics.run!(sampler, model, at, 1000; outp=nothing)
        # push data with Gaussian noise
        Energy = energy(fsmodel, at)
        Forces = forces(fsmodel, at) 
        NoisyEnergy = Energy + rand(Normal(0, energy_std))
        NoisyForces = [ f + ACE.DState(rr = forces_std * randn(SVector{3, Float64})) for f in Forces]
        push!( data, (at= deepcopy(at), E = NoisyEnergy, F= NoisyForces) )
        println("Data added: ", k)
    end

    theta = params(model)

    return (theta, data)
end 

Theta = load_object("Data/dataset1.jld2")["theta"];
Data = load_object("Data/dataset1.jld2")["data"];

function log_likelihood_L1(pot, d; ωE = 1.0, ωF = 1.0/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * abs(d.E - Energy(pot, d.at)) -  ωF * sum(sum(abs2, g - f.rr) 
                     for (g, f) in zip(Forces(pot, d.at), d.F))
end 

function GaussianPrior(pot) 
    # Gaussian prior 
    return MvNormal(zeros(length(get_params!(pot))),I)
end 


statModel1 = StatisticalModel(log_likelihood_L1, GaussianPrior, pot, Data); 

log_likelihood(statModel1)

get_gll(statModel1)


function log_posterior(sm::StatisticalModel; mb_size::Int = 0)
    if mb_size == 0 || mb_size >= length(sm.data)
        # calculates the log posterior for entire dataset contained in sm 
        prior_value = logpdf(sm.prior(sm.model), reshape(get_params!(sm.model), length(get_params!(sm.model))))     # this reshape may cause a bit of a bug 
        return prior_value + sum(sm.log_likelihood(sm.model, d) for d in sm.data)
    else 
        # calculates log posterior for mini-batch of dataset 
        data_size = length(sm.data) 
        mbatch_index = sample(1:data_size, mb_size, replace = false)
        mbatch = [sm.data[i] for i in mbatch_index]
        prior_value = logpdf(sm.prior(sm.model), reshape(get_params!(sm.model), length(get_params!(sm.model))))     # this reshape may cause a bit of a bug 

        return prior_value + (data_size / mb_size) * sum(sm.log_likelihood(sm.model, d) for d in mbatch)
    end 
    
end 

log_posterior(statModel1)

function U(sm::StatisticalModel, θ; mb_size::Int = 0)
    if length(θ) != length(get_params!(sm.pot))
        error("The number of parameters does not match")
    end 

    if mb_size == 0 || mb_size >= length(sm.data) 
        set_params!(sm.pot, θ)
        return - log_posterior(sm)
    else 
        set_params!(sm.pot, θ)
        return - log_posterior(sm; mb_size)
    end 
    
end 

p = rand(30)
U(statModel1, p; mb_size=10) # A potential function defined over R^30



gradU(statModel1, rand(30))

mutable struct Stateθ
    θ
    θ_prime 
end 

mutable struct BAOABθ
    h::Float64      # Step size
    F::Vector{SVector{3, Float64}}
    β::Float64 
    γ::Float64 
    sm::StatisticalModel
end 
# BAOABθ(h::T, state::Stateθ, sm ; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOABθ(h, gradU(sm, state.θ), β, γ, sm)
BAOABθ(h::T, state::Stateθ, sm ; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOABθ(h, randn(ACE.SVector{3,Float64}, length(st.θ_prime)) , β, γ, sm)


# function step!(s::BAOABθ, st::Stateθ) 
#     st.θ_prime += 0.5 * s.h * s.F
#     st.θ += 0.5 * s.h * st.θ_prime 
#     st.θ_prime = exp(-s.h * s.γ) * st.θ_prime + sqrt((1/s.β) * (1 - exp(-2*s.γ*s.h))) * randn(ACE.SVector{3,Float64}, length(st.θ_prime)) 
#     st.θ += 0.5 * s.h * st.θ_prime
#     # s.F = gradU(s.sm, st.θ)
#     s.F = randn(ACE.SVector{3,Float64}, length(st.θ_prime)) 
#     st.θ_prime += 0.5 * s.h * s.F
# end 

# function run!(st::Stateθ, sampler::BAOABθ, Nsteps::Int64, outp) 
#     push!(outp[1], st.θ)
#     push!(outp[2], st.θ_prime)
#     for _ in 1:Nsteps 
#         step!(sampler, st) 
#         push!(outp[1], st.θ)
#         push!(outp[2], st.θ_prime)
#     end 
# end 

st = Stateθ(randn(ACE.SVector{3,Float64}, 30), randn(ACE.SVector{3,Float64}, 30))
sampler = BAOABθ(0.1, st, statModel1)
outp = [[], []] 
st.θ_prime + sampler.F
run!(st, sampler, 100, outp) 
plot(1:101, [elem[1][3] for elem in outp[1]])
outp[1][1]








# outp = BayesianMLIP.MHoutputschedulers.MHoutp()
# # MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(true_θ)
# MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(rand(length(params(fsmodel))))
# BayesianMLIP.Samplers.run!(MetroHastings, statModel, 40, outp)


# timesteps = 500
# no_trials = 30

# outp_collection = [ BayesianMLIP.MHoutputschedulers.MHoutp() for i in 1:no_trials] 
# for i in 1:no_trials 
#     MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(rand(length(params(fsmodel))))
#     BayesianMLIP.Samplers.run!(MetroHastings, statModel, timesteps, outp_collection[i])
# end 

# comb = log_posterior(statModel, Theta) * ones(timesteps)
# for i in 1:no_trials
#     comb = hcat(comb, outp_collection[i].log_posterior)
# end 

# plot(1:timesteps, comb, title="MH TimeStep vs Log Posterior Value", labels=nothing)
