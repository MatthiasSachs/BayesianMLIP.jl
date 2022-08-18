using ACE, ACEatoms, Plots, ACEflux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Distributions
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics  
using BayesianMLIP.MiniACEflux, BayesianMLIP.Utils, BayesianMLIP.Samplers, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser
using ACEflux: FluxPotential
import Distributions: logpdf, MvNormal
using JSON
using Flux
# Initialize Finnis-Sinclair Model with ACE basis (w/ coefficients=0)
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 2), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 6.0); 

# Initialize log-likelihood to be 0
log_likelihood_toy = nothing
# (pot::ACEflux.FluxPotential, d) -> -0.0

# Set prior to be Gaussian with some arbitrary Sigma 
X = rand(30, 30)
Sigma = X' * X
priorNormal_toy = MvNormal(zeros(30), Sigma)

get_glpr(stm::StatisticalModel)
# Generate dummy dataset 
Data = zeros(100)

# Initialize StatisticalModel object
stm2 = StatisticalModel(log_likelihood_toy, priorNormal_toy, pot, Data); 

# Check that log_likelihood of entire dataset is 0
log_likelihood(stm2)
log_prior(stm2)
log_posterior(stm2)

# Initialize state to some random position and random momentum
st_toy = State_θ(randn(30), randn(30))

# Initialize outputscheduler recording θ (position), θ' (momenta), and log_posterior values 
BAOABoutp = BAOABoutp_θ()  

# Construct sampler with step size 0.001 and miniBatch size of 1. 
# This is where the bug is: "no method matching vec(::Nothing)" 
BAOABsampler = BAOAB_θ(0.1, st_toy, stm2, 1; β=1.0, γ=10.0); 

# Running over 5000 steps 
Samplers.run!(st_toy, BAOABsampler, stm2, 5000, BAOABoutp)
