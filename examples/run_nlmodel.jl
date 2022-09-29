using ACE, ACEatoms, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics
import StatsBase: sample
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics  
using BayesianMLIP.MiniACEflux, BayesianMLIP.Utils, BayesianMLIP.Samplers, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser
using Random: seed!, rand
import BayesianMLIP.NLModels: Hamiltonian, energy, forces, get_params, set_params!
using ACEflux: FluxPotential
import Distributions: logpdf, MvNormal
using JSON

at = bulk(:Cu, cubic=true) * 3; rattle!(at, 0.1) ; 
rcut = 3.0

# Initialize Finnis-Sinclair Model 
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
model = Chain(Linear_ACE(;ord = 2, maxdeg = 3, Nprop = 2, rcut=rcut), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, rcut); 

get_params(pot)
set_params!(pot, randn(nparams(pot)))


function log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = ωE/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) 
                     for (g, f) in zip(forces(pot, d.at), d.F))
end 

log_likelihood_Null = ConstantLikelihood() 

# Generate positive definite matrix 
n = nparams(pot)
a = randn(n, n); A = a' * a; A = (A' + A)/2; eigen(A).values
priorNormal = MvNormal(zeros(nparams(pot)), [A zeros(n, n); zeros(n, n) 1e-2 * Diagonal(ones(n))])
priorUniform = FlatPrior()

# real data 
real_data = getData(JSON.parsefile("/z1-mbahng/mbahng/mlearn/data/Cu/training.json"))[1:10] ; # 262-vector 


stm1 = StatisticalModel(log_likelihood_Null, priorNormal, pot, real_data) ;

# Precision matrix, covariance matrix, and mean of posterior Gaussian calculated analytically 
linear_post_mean_cov = precon_pre_cov_mean(stm1)

# If we would like to make nonlinear preconditioning, then we just set it as a block matrix [Σ 0; 0 0]
precon_precision = linear_post_mean_cov["true_precision"]
precon_covariance = linear_post_mean_cov["true_covariance"]
eigen(precon_precision).values

μ_posterior = vcat(linear_post_mean_cov["true_mean"], zeros(nlinparams(stm1.pot)))
# true_μ_posterior = load("true_mu_posterior.jld2")["empirical_true_mean"]

st1 = State_θ(zeros(nparams(pot)), zeros(nparams(pot))) ;
AMHoutp1 = MHoutp_θ() ; 
AMHsampler1 = AdaptiveMHsampler(1., st1, stm1, st1.θ, A) ; 
Samplers.run!(st1, AMHsampler1, stm1, 10000, AMHoutp1)
Histogram(AMHoutp1)
Trajectory(AMHoutp1) 
Summary(AMHoutp1)

Histogram(AMHoutp1; save_fig=true, title="NonLin_LinHist")
Trajectory(AMHoutp1; save_fig=true, title="NonLin_LinTraj")
Summary(AMHoutp1; save_fig=true, title="NonLin_LinSumm")

dict = Dict{String, Any}("st1" => st1, 
                         "AMHsampler1"  => AMHsampler1, 
                         "stm1" => stm1, 
                         "AMHoutp1" => AMHoutp1)
save("continue_running.jld2", dict)