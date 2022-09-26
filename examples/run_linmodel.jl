using ACE, ACEatoms, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics
import StatsBase: sample
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics  
using BayesianMLIP.MiniACEflux, BayesianMLIP.Utils, BayesianMLIP.Samplers, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser
using Random: seed!, rand
using ACEflux: FluxPotential
import Distributions: logpdf, MvNormal
using JSON


at = bulk(:Cu, cubic=true) * 3; rattle!(at, 0.1) ; 
rcut = 3.0

# Initialize linear model
FS(ϕ) = ϕ[1]
model = Chain(Linear_ACE(;ord = 2, maxdeg = 1, Nprop = 1, rcut=rcut), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, rcut); 

get_params(pot)
set_params!(pot, randn(nparams(pot))) 

function log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = ωE/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) 
                     for (g, f) in zip(forces(pot, d.at), d.F))
end 

log_likelihood_Null = ConstantLikelihood() 

a = randn(nparams(pot), nparams(pot)); A = a' * a; A = (A' + A)/2; ev = eigen(A).values; println(maximum(ev)/minimum(ev))
A
priorNormal = MvNormal(zeros(nparams(pot)), A)
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

μ_posterior = linear_post_mean_cov["true_mean"]
eigen(A).values

st1 = State_θ(zeros(nparams(pot)), zeros(nparams(pot))) ;
AMHoutp1 = MHoutp_θ() ; 
AMHsampler1 = AdaptiveMHsampler(1e-1, st1, stm1, st1.θ, A) ; 
Samplers.run!(st1, AMHsampler1, stm1, 30000, AMHoutp1)
Histogram(AMHoutp1) 
Trajectory(AMHoutp1) 
Summary(AMHoutp1)

Histogram(AMHoutp1; save_fig=true, title="LinHist_CNum55")
Trajectory(AMHoutp1; save_fig=true, title="LinTraj_CNum55")
Summary(AMHoutp1; save_fig=true, title="LinSumm_CNum55")


dict = Dict{String, Any}("st1" => st1, 
                         "AMHsampler1"  => AMHsampler1, 
                         "stm1" => stm1, 
                         "AMHoutp1" => AMHoutp1)
save("linear_run.jld2", dict)


# Constuct BADODAB sampler and run
st = State_θ(rand(30), zeros(30))
BADODABoutp = BADODABoutp_θ()
BADODABsampler = BADODAB_θ(1.0e-15, st, stm1, 1; β=1.0, μ=0.001, ξ=300.0, σG=0.0, σA=10.0); 
Samplers.run!(st, BADODABsampler, stm1, 2000, BADODABoutp) 
θ_traj2 = [θ[8] for θ in BADODABoutp.θ]
plot(1:length(θ_traj2), θ_traj2, title="Component Position Trajectory", legend=false)
# histogram(θ_traj2, bins = :scott, title="")




BADODABsampler = BADODAB_θ(0.0001, st, stm1, 1; β=Inf, μ=1000000.0, ξ=0.0, σG=1.0, σA=10.0)
BADODABsampler.F
BADODABoutp = [[], [], []] 
Nsteps = 100
run!(st, BADODABsampler, stm1, Nsteps, BADODABoutp) 
θ_traj2 = [θ[1] for θ in BADODABoutp[1]]
plot(1:length(θ_traj2), θ_traj2, title="Component Position Trajectory", legend=false)
# histogram(θ_traj2, bins = :scott, title="")

θ_prime_traj2 = [θ[1] for θ in BADODABoutp[2]]
plot(1:length(θ_prime_traj2), θ_prime_traj2, legend=false)
histogram(θ_prime_traj2, bins = :scott, title="")

energy_traj2 = [energy for energy in BADODABoutp[3]]
plot(1:length(energy_traj2), energy_traj2, title="Log-Posterior Value", legend=false)