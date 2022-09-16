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
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 1, rcut=rcut), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, rcut); 

get_params(pot)
set_params!(pot, randn(1, 15)) 
nparams(pot)


function log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = ωE/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) 
                     for (g, f) in zip(forces(pot, d.at), d.F))
end 

log_likelihood_Null = ConstantLikelihood() 

priorNormal = MvNormal(zeros(nparams(pot)),I)
priorUniform = FlatPrior()

# real data 
real_data = getData(JSON.parsefile("/z1-mbahng/mbahng/mlearn/data/Cu/training.json"))[1:10] ; # 262-vector 

# artificial data 
info = load("./Run_Data/Artificial_Data/artificial_data_Noisy_30.jld2")
artificial_data = info["data"][1:3:end]
true_θ = () -> load("./Run_Data/Artificial_Data/artificial_data_Noisy_30.jld2")["theta"]
true_θ()


stm1 = StatisticalModel(log_likelihood_L2, priorUniform, pot, real_data) ;
Ψ = design_matrix(stm1)
Y = get_Y(stm1)
Σ_0 = I
Σ_Tilde = get_Σ_Tilde(stm1)
β = 1.0 

Σ_posterior = 1 ./ (Σ_0 + β * transpose(Ψ) * Σ_Tilde * Ψ)
μ_posterior = β * Σ_posterior * transpose(Ψ) * Y


ev = eigen(Σ_posterior).values
maximum(ev)/minimum(ev)

# Run AMH sampler initialized at random point with isotropic preconditioning matrix 

st1 = State_θ(μ_posterior, zeros(nparams(pot))) ;
st1 = State_θ(randn(15), zeros(nparams(pot))) ;
AMHoutp1 = MHoutp_θ() ; 
AMHsampler1 = AdaptiveMHsampler(1e-5, st1, stm1, st1.θ, Σ_posterior) ;  
Samplers.run!(st1, AMHsampler1, stm1, 1000, AMHoutp1)
Samplers.run!(st1, AMHsampler1, stm1, 3000, AMHoutp1)
Histogram(AMHoutp1)
Trajectory(AMHoutp1)
Summary(AMHoutp1)


# Diagonal matrix: variance of each basis evaluation over all atomic environments of all data 
precon_params = get_precon_params(stm1)[:,2];
precon_params[1] = 1.;
precon_params = 1 ./ precon_params;
Σ_initial = Diagonal(precon_params);


# Run same but with preconditioned matrix according to empirical covariance 
st2 = approx_minimum;
AMHoutp2 = MHoutp_θ()    
AMHsampler2 = AdaptiveMHsampler(1e-20, st2, stm1, st2.θ, Σ_initial) ;  
Samplers.run!(st2, AMHsampler2, stm1, 3000, AMHoutp2)
Samplers.run!(st2, AMHsampler2, stm1, 3000, AMHoutp2)
Histogram(AMHoutp2)
Trajectory(AMHoutp2)
Summary(AMHoutp2)




# Initialize StatisticalModel 
s = load("continue_running.jld2")
stm1 = s["stm1"]
st1 = s["st1"]
AMHoutp1 = s["AMHoutp1"]
AMHsampler1 = s["AMHsampler1"]
stm1 = StatisticalModel(log_likelihood_L2, priorUniform, pot, real_data)
st1 = State_θ(reshape(true_θ(), 15), zeros(nparams(stm1.pot)))
get_precon_params(stm1)

precon_params = get_precon_params(stm1)[:,2]
precon_params[1] = 1.
precon_params = 1 ./ precon_params
Σ_initial = Diagonal(precon_params)

AMHoutp1 = MHoutp_θ()    
AMHsampler1 = AdaptiveMHsampler(1e-4, st1, stm1, st1.θ, Σ_initial) ;  
Samplers.run!(st1, AMHsampler1, stm1, 500, AMHoutp1)


# AMH Sampler 




AMHoutp1 = s["AMHoutp1"]

AMHsampler1 = s["AMHsampler1"]

Samplers.run!(st1, AMHsampler1, stm1, 10, AMHoutp1)
# runs 
dict = Dict{String, Any}("st1" => st1, 
                         "AMHsampler1"  => AMHsampler1, 
                         "stm1" => stm1, 
                         "AMHoutp1" => AMHoutp1)
save("continue_running.jld2", dict)




AMHsampler1.Σ
eigen(AMHsampler1.Σ).values

Histogram(AMHoutp1)
Trajectory(AMHoutp1)
Summary(AMHoutp1)
Histogram(AMHoutp1; save_fig=true, title="AMH_Hist_1e-3_Preconditioned2")
Trajectory(AMHoutp1; save_fig=true, title="AMH_Traj_1e-3_Preconditioned2")
Summary(AMHoutp1; save_fig=true, title="AMH_Summ_1e-3_Preconditioned2")




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