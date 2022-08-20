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
using Flux 

# Initialize Finnis-Sinclair Model with ACE basis (w/ coefficients=0)
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/9) - 1/3
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 2), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 3.0); 

# Initialize log-likelihood to be 0
log_likelihood_toy = ConstantLikelihood()

# Set prior to be Gaussian with some Sigma 
dim = nparams(pot)
X = rand(dim, dim)
Sigma = X' * X
k = eigen(Sigma).values
Sigma = Diagonal(k)
priorNormal_toy = MvNormal(zeros(dim), Sigma)


eigen(Sigma).values 

eigen(Sigma).values[end] / eigen(Sigma).values[1]

# Generate dummy dataset 
Data = zeros(100);

stm2 = StatisticalModel(log_likelihood_toy, priorNormal_toy, pot, Data); 

ll = get_ll(stm2) 
lpr = get_lpr(stm2)

lpr(zeros(dim))
ll(zeros(dim), Data[1])

# Run Adaptive Metropolis Hastings 
st = State_θ(zeros(nparams(stm2.pot)), zeros(nparams(stm2.pot)))
AMHoutp = MHoutp_θ()    # new run 
AMHsampler = AdaptiveMHsampler(1.0, st, stm2, st.θ, I) ;    
Samplers.run!(st, AMHsampler, stm2, 100000, AMHoutp; trueΣ=Sigma)

Histogram(AMHoutp; save_fig=false)
Trajectory(AMHoutp; save_fig=false)
Summary(AMHoutp; save_fig=false)

Histogram(AMHoutp; save_fig=true)
Trajectory(AMHoutp; save_fig=true)
Summary(AMHoutp; save_fig=true)

# Run BAOAB 
st_toy = State_θ(randn(30), randn(30))
BAOABoutp = BAOABoutp_θ()      
BAOABsampler = BAOAB_θ(0.001, st_toy, stm2, 1; β=1.0, γ=10.0); 

Samplers.run!(st_toy, BAOABsampler, stm2, 5000, BAOABoutp)

plotTrajectory(BAOABoutp, 1)
histogramTrajectory(BAOABoutp,10)
plotLogPosterior(BAOABoutp)

get_glpr(stm2)(st_toy.θ)