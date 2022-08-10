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
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 2), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 6.0);
# Don't need to initialize this since it'll be assigned in run, but for testing purposes 
set_params!(pot, randn(30))         

# Initialize atomic configuration
at = bulk(:Cu, cubic=true) * 3;
rattle!(at, 0.1) ;       # If we rattle this too much, the integrators can become unstable

# Testing parameters (NLModels), energy/forces (JuLIP) functions 
get_params(pot)
nparams(pot)
energy(pot, at)
forces(pot, at)

# Initialize log-likelihood to be 0
log_likelihood_toy = (pot::ACEflux.FluxPotential, d) -> -0.0

# Set prior to be Gaussian with some Sigma 
X = rand(30, 30)
Sigma = X' * X
priorNormal_toy = MvNormal(zeros(30), Sigma)

# Generate dummy dataset 
at_dataset = JSON.parsefile("./train_test/Cu/training.json"); # 262-vector 
Data = getData(at_dataset)

stm2 = StatisticalModel(log_likelihood_toy, priorNormal_toy, pot, Data); 
set_params!(stm2.pot, zeros(30))
log_likelihood(stm2)
log_prior(stm2)
log_posterior(stm2)

st_toy = State_θ(randn(30), randn(30))

# Run Adaptive Metropolis Hastings 
AMHoutp = MHoutp_θ()
AMHsampler = AdaptiveMHsampler(0.5, st_toy, stm2, st_toy.θ, I)
Samplers.run!(st_toy, AMHsampler, stm2, 100000, AMHoutp)

plotRejectionRate(AMHoutp)
plotTrajectory(AMHoutp, 1)
histogramTrajectory(AMHoutp, 1) 
true_eigen_ratio = eigen(Sigma).values[30]/eigen(Sigma).values[1]
plotEigenRatio(AMHoutp)



# Run BAOAB 
st_toy = State_θ(randn(30), randn(30))
BAOABoutp = BAOABoutp_θ()      
BAOABsampler = BAOAB_θ(0.001, st_toy, stm2, 1; β=1.0, γ=10.0); 

Samplers.run!(st_toy, BAOABsampler, stm2, 5000, BAOABoutp)

plotTrajectory(BAOABoutp, 1)
histogramTrajectory(BAOABoutp,10)
plotLogPosterior(BAOABoutp)

get_glpr(stm2)(st_toy.θ)