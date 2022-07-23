using ACE, ACEatoms, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2, JuLIP, Plots, StaticArrays
import StatsBase: sample
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics  
using BayesianMLIP.MiniACEflux, BayesianMLIP.Utils      
using Random: seed!, rand
using ACE: O3, evaluate, val, SymmetricBasis, State
import BayesianMLIP.NLModels: gradParams, Hamiltonian
# using BayesianMLIP.MiniACEflux: FluxPotential
using ACEflux: FluxPotential
using Distributions
import Distributions: logpdf, MvNormal


# Initialize Finnis-Sinclair Model with ACE basis (w/ coefficients=0)
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 2), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 6.0);
θ = randn(30)
set_params!(pot, θ)

# Initialize atomic configuration
at = bulk(:Cu, cubic=true) * 3 ;
rattle!(at,0.6) ;
# at2 = deepcopy(at) ;


# Testing parameters (NLModels), energy/forces (JuLIP) functions 
get_params(pot)
nparams(pot)
energy(pot, at)
forces(pot, at)


# Run BAOAB to sample from Gibbs measure of Langevin dynamics with ACE potential
BAOAB_Sampler = BAOAB(0.1, pot, at; γ=1.0, β=1.0)
BAOsteps = []
run!(BAOAB_Sampler, pot, at, 200; outp = BAOsteps) 
x_traj1 = [step.X[1][1] for step in BAOsteps]
plot(1:length(BAOsteps), x_traj1, title="BAOAB Component Trajectory", legend=false)
histogram(x_traj1, bins = :scott, title="Histogram of BAOAB Samples", legend=false)
plot(1:length(x_traj), [Hamiltonian(pot, elem) for elem in BAOsteps], title="BAOAB Hamiltonian")

# Run BADODADB 
BADODAB_Sampler = BADODAB(0.1, pot, at)
BADODABsteps = []
run!(BADODAB_Sampler, pot, at, 500; outp = BADODABsteps)
x_traj2 = [step.X[1][1] for step in BADODABsteps]
plot(1:length(BADODABsteps), x_traj2, title="BADODAB Component Trajectory")
histogram(x_traj2, bins = :scott, title="Histogram of BADODAB Samples")
plot(1:length(BADODABsteps), [Hamiltonian(pot, elem) for elem in BADODABsteps], title="BADODAB Hamiltonian")

# Load data, define log_likelihood and prior
# Theta = load_object("Data/dataset1.jld2")["theta"];
# Data = load_object("Data/dataset1.jld2")["data"];

function log_likelihood_L1(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = 1.0/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * abs(d.E - energy(pot, d.at)) -  ωF * sum(sum(abs2, g - f) 
                     for (g, f) in zip(forces(pot, d.at), d.F))
end 

priorNormal = MvNormal(zeros(30),I)

function generate_data(Ndata::Int64, sampler, pot::ACEflux.FluxPotential, at::AbstractAtoms) 
    data = [] 
    for i in 1:Ndata 
        run!(sampler, pot, at, 1000; outp=nothing) 
        # push data (without Gaussian noise for now) 
        Energy = energy(pot, at) 
        Forces = forces(pot, at) 
        push!(data, (at=deepcopy(at), E = Energy, F = Forces))
        println("Data added: ", i)
    end 

    theta = get_params(pot) 

    return (theta, data) 
end 

Theta, Data = generate_data(2, BADODAB_Sampler, pot, at)

# Initialize StatisticalModel 
stm1 = StatisticalModel(log_likelihood_L1, priorNormal, pot, Data); 

# Testing params and nparams functions (Utils)
Flux.params(model).params.dict.ht
Flux.params(pot)
Flux.params(stm1)


log_likelihood(stm1)
log_prior(stm1)
log_posterior(stm1)


# Compute gradient functions 
using BayesianMLIP.Utils: get_glp, get_glpr, get_gll
gll = get_gll(stm1)
glpr = get_glpr(stm1)
glp = get_glp(stm1)

# The function U = -LogPosterior doesn't actually have to be explictly defined. We only need computation of -∇U.
# Since glp = ∇LogPosterior ⟹ glp = -∇U


# Implement BAOAB with mini-batch 
data_size = length(stm1.data)
mb_size = 1
mbatch_indices = sample(1:data_size, mb_size, replace = false)
miniBatch = [stm1.data[i] for i in mbatch_indices] ;

mutable struct State_θ
    θ
    θ_prime 
end 

abstract type param_Integrator end 

mutable struct BAOAB_θ <: param_Integrator
    h::Float64      # Step size
    F::Vector{Float64}
    β::Float64 
    γ::Float64 
end 
BAOAB_θ(h::T, state::State_θ ; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOAB_θ(h, glp(state.θ, miniBatch, data_size), β, γ)

mutable struct BADODAB_θ <: param_Integrator
    h::Float64 
    F::Vector{Float64}
    β::Float64 
    n::Int64
    σG::Float64
    σA::Float64
    μ::Float64
    ξ::Float64
end 

BADODAB_θ(h::Float64, state::State_θ; β=1.0, n=length(state.θ), σG=1.0, σA=9.0, μ=10.0, ξ=1.0) = BADODAB_θ(h, glp(state.θ, miniBatch, data_size), β, n, σG, σA, μ, ξ)


function step!(s::BAOAB_θ, st::State_θ) 
    st.θ_prime += 0.5 * s.h * s.F
    st.θ += 0.5 * s.h * st.θ_prime 
    st.θ_prime = exp(-s.h * s.γ) * st.θ_prime + sqrt((1/s.β) * (1 - exp(-2*s.γ*s.h))) * randn(length(st.θ_prime)) 
    st.θ += 0.5 * s.h * st.θ_prime
    s.F = glp(st.θ, miniBatch, data_size)
    st.θ_prime += 0.5 * s.h * s.F
end 

function step!(s::BADODAB_θ, st::State_θ) 
    st.θ_prime += 0.5 * s.h * (s.F + s.σG .* randn(length(st.θ)))
    st.θ += 0.5 * s.h * st.θ_prime 
    s.ξ += 0.5 * s.h * (1/s.μ) * (dot(st.θ_prime, st.θ_prime) - s.n * (1/s.β))

    if s.ξ == 0.0 
        st.θ_prime = st.θ_prime + sqrt(s.h) * s.σA .* randn(length(st.θ))
    else 
        α = 1 - exp(-2 * s.ξ * s.h) 
        ζ = 2 * s.ξ
        st.θ_prime = exp(-s.ξ * s.h) * st.θ_prime + s.σA * sqrt(α/ζ) * randn(length(st.θ))
    end 

    s.ξ += 0.5 * s.h * (1/s.μ) * (dot(st.θ_prime, st.θ_prime) - s.n * (1/s.β))
    st.θ += 0.5 * s.h * st.θ_prime 
    s.F = glp(st.θ, miniBatch, data_size)
    st.θ_prime += 0.5 * s.h * (s.F + s.σG .* randn(length(st.θ)))
end 

import BayesianMLIP.Dynamics: run!

function run!(st::State_θ, sampler, Nsteps::Int64, outp) 
    push!(outp[1], st.θ)
    push!(outp[2], st.θ_prime)
    println(0, "/", Nsteps)
    for i in 1:Nsteps 
        step!(sampler, st) 
        push!(outp[1], st.θ)
        push!(outp[2], st.θ_prime)
        println(i, "/", Nsteps)
    end 
end 

# Initialize state in random vector of R^30 × R^30
st = State_θ(randn(30), randn(30))

# Construct BAOAB sampler and run
BAOABsampler = BAOAB_θ(0.001, st)
BAOABoutp = [[], []] 
Nsteps = 100
run!(st, BAOABsampler, Nsteps, BAOABoutp) 

# Constuct BADODAB sampler and run
BADODABsampler = BADODAB_θ(0.000001, st)
BADODABoutp = [[], []] 
Nsteps = 100
run!(st, BADODABsampler, Nsteps, BADODABoutp) 


θ_traj = [elem[1] for elem in BADODABoutp[1]]
plot(1:length(θ_traj), θ_traj)
histogram(θ_traj, bins = :scott, title="")
get_params(stm1.pot)

 
println("end")





# outp = BayesianMLIP.MHoutputschedulers.MHoutp()
# # MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(true_θ)
# MetroHastings = BayesianMLIP.Samplers.SimpleMHsampler(rand(length(params(fsmodel))))
# BayesianMLIP.Samplers.run!(MetroHastings, statModel, 40, outp)


# timesteps = 500
# no_trials = 30

# outp_collection = [ BayesianMLIP.MHoutputschedulers.MHoutp() for i in 1:no_trials] 
# for i in 1:no_trials 
#     MetroHastings = BayesianMLIP.NstepsSamplers.SimpleMHsampler(rand(length(params(fsmodel))))
#     BayesianMLIP.Samplers.run!(MetroHastings, statModel, timesteps, outp_collection[i])
# end 

# comb = log_posterior(statModel, Theta) * ones(timesteps)
# for i in 1:no_trials
#     comb = hcat(comb, outp_collection[i].log_posterior)
# end 

# plot(1:timesteps, comb, title="MH TimeStep vs Log Posterior Value", labels=nothing)
