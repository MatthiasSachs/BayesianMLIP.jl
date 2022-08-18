using ACE, ACEatoms, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2, JuLIP, StaticArrays
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

basis = Linear_ACE(;ord = 2, maxdeg =4, Nprop = 2).m.basis
scaling = ACE.scaling(basis, 2)



function log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = 1.0/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) 
                     for (g, f) in zip(forces(pot, d.at), d.F))
end 

function log_likelihood_0(pot::ACEflux.FluxPotential, d)
    return 0.0 
end 

priorNormal = MvNormal(zeros(nparams(pot)),I)

priorUniform =  1.0         # how to create distribution object? 

# real data 
real_data = getData(JSON.parsefile("./Run_Data/Real_Data/training_test/Cu/training.json")) # 262-vector 

# artificial data 
info = load("./Run_Data/Artificial_Data/artificial_data2.jld2")
artificial_data = info["data"][1:end]
true_θ = () -> load("./Run_Data/Artificial_Data/artificial_data2.jld2")["theta"]

# Initialize StatisticalModel 
stm1 = StatisticalModel(log_likelihood_L2, priorNormal, pot, artificial_data); 

set_params!(stm1.pot, zeros(nparams(stm1.pot)))
@time log_likelihood(stm1)
@time log_prior(stm1)
@time log_posterior(stm1)

set_params!(stm1.pot, true_θ())

log_likelihood(stm1)
log_prior(stm1)
log_posterior(stm1)


# SLGD Sampler 
true_θ()
st = State_θ(reshape(true_θ(), nparams(stm1.pot)), zeros(nparams(stm1.pot)))
SGLDoutp = SGLDoutp_θ() 
SGLDsampler = SGLD_θ(1e-7, st, stm1, 1; β=1.) ;
Samplers.run!(st, SGLDsampler, stm1, 30000, SGLDoutp) 
Trajectory(SGLDoutp)

plotTrajectory(SGLDoutp, 3)
histogramTrajectory(SGLDoutp, 4)
plotLogPosterior(SGLDoutp)


# AMH Sampler 
st = State_θ(reshape(true_θ(), nparams(stm1.pot)), zeros(nparams(stm1.pot)))
AMHoutp = MHoutp_θ()    # new run 
AMHsampler = AdaptiveMHsampler(0.03, st, stm1, st.θ, I)     
Samplers.run!(st, AMHsampler, stm1, 50000, AMHoutp)

Histogram(AMHoutp)
Trajectory(AMHoutp)
Summary(AMHoutp)


# Save last state, last state of sampler, statistical model, and outp to jld2 file 
dict = Dict{String, Any}( "description" => :"10000 AMH steps run on Artificial Data [1:2:500] initialized at mode", 
                          "state" => st, 
                          "sampler" => AMHsampler, 
                          "stm" => stm1, 
                          "outp" => AMHoutp, 
                          "results" => "Step size too small and acceptance rate too large. ")


save("./Run_Data/Artificial_Data/AdaptiveMetropolisHastings/AMH_4.jld2", dict) 

# Construct sampler and run
info = load("./Run_Data/AMH_Cu_Training1/AMH_Cu_Training1-1081.jld2")
st = info["state"]      # stable θ
BAOABoutp = BAOABoutp_θ()        # θ, θ', log_posterior 
BAOABsampler = BAOAB_θ(1e-4, st, stm1, 1; β=1., γ=1e7); 
Samplers.run!(st, BAOABsampler, stm1, 1000, BAOABoutp)

plotMomenta(BAOABoutp, 6)
plotTrajectory(BAOABoutp, 1) 
histogramTrajectory(BAOABoutp, 5) 
plotLogPosterior(BAOABoutp)

info2 = load("./Run_Data/AMH_Cu_Training1/AMH_Cu_Training1-1081.jld2")
st2 = info["state"]      # stable θ
BAOABoutp2 = BAOABoutp_θ()        # θ, θ', log_posterior 
BAOABsampler2 = BAOAB_θ(1e-6, st2, stm1, 1; β=1., γ=1e7); 
Samplers.run!(st2, BAOABsampler2, stm1, 1000, BAOABoutp2)

plot(1:1000, [elem[1] for elem in BAOABoutp2.θ])

plotMomenta(BAOABoutp2, 6)
plotTrajectory(BAOABoutp2, 2) 
histogramTrajectory(BAOABoutp2, 4) 
plotLogPosterior(BAOABoutp2)


# Constuct BADODAB sampler and run
st = State_θ(rand(30), zeros(30))
BADODABoutp = BADODABoutp_θ()
BADODABsampler = BADODAB_θ(1.0e-15, st, stm1, 1; β=1.0, μ=0.001, ξ=300.0, σG=0.0, σA=10.0); 
Samplers.run!(st, BADODABsampler, stm1, 2000, BADODABoutp) 
θ_traj2 = [θ[8] for θ in BADODABoutp.θ]
plot(1:length(θ_traj2), θ_traj2, title="Component Position Trajectory", legend=false)
# histogram(θ_traj2, bins = :scott, title="")

θ_prime_traj2 = [θ[1] for θ in BADODABoutp.θ_prime]
plot(1:length(θ_prime_traj2), θ_prime_traj2, legend=false)
histogram(θ_prime_traj2, bins = :scott, title="")

energy_traj2 = [energy for energy in BADODABoutp.log_posterior]
plot(1:length(energy_traj2), energy_traj2, title="Log-Posterior Value", legend=false)
Theta

plot(1:length(θ_traj2), BADODABoutp.ξ, title="ξ Value", legend=false)


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