using ACE, ACEatoms, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics
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
FS(ϕ) = ϕ[1] #+ sqrt(abs(ϕ[2]) + 1/9) - 1/3
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 1), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 3.0); 
nparams(pot)
get_params(pot)
# basis = Linear_ACE(;ord = 2, maxdeg =4, Nprop = 2).m.basis
# scaling = ACE.scaling(basis, 2)


function log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = 1.0/(3*length(d.at)) )
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
info = load("./Run_Data/Artificial_Data/artificial_data1.jld2")
artificial_data = info["data"][1:end]
true_θ = () -> load("./Run_Data/Artificial_Data/artificial_data1.jld2")["theta"]
true_θ()

# Initialize StatisticalModel 
stm1 = StatisticalModel(log_likelihood_L2, priorUniform, pot, real_data); 
get_precon_params(stm1)
precon_params = get_precon_params(stm1)[:,2]
precon_params[1] = 1e-6
precon_params = 1 ./ precon_params
Σ_initial = Diagonal(precon_params)

ll = get_ll(stm1) ;
lpr = get_lpr(stm1) ;
lp = get_lp(stm1) ;

# SGLD Sampler 
# st = State_θ(reshape(true_θ(), nparams(stm1.pot)), zeros(nparams(stm1.pot)))
st = State_θ(reshape(true_θ(), nparams(stm1.pot)) + 0.01 * randn(nparams(stm1.pot)), zeros(nparams(stm1.pot)))
SGLDoutp = SGLDoutp_θ()  
SGLDsampler = SGLD_θ(1e-11, st, stm1, 2; β=Inf) ;
@time Samplers.run!(st, SGLDsampler, stm1, 30, SGLDoutp)  

plot([elem[1] for elem in SGLDoutp.θ], [elem[2] for elem in SGLDoutp.θ], legend=false)

outps = []
for n_samplers in 1:10
    st = State_θ(reshape(true_θ(), nparams(stm1.pot)) + 0.1 * randn(nparams(stm1.pot)), zeros(nparams(stm1.pot)))
    SGLDoutp = SGLDoutp_θ()  
    SGLDsampler = SGLD_θ(1e-11, st, stm1, 10; β=Inf) ;
    Samplers.run!(st, SGLDsampler, stm1, 100, SGLDoutp) 
    
    push!(outps, SGLDoutp)
end 

p = scatter([true_θ()[1]], [true_θ()[2]], markershape=:star, markersize=7, color=:red)
for outp in outps
    plot!([t[1] for t in outp.θ], [t[2] for t in outp.θ], legend=false, markershape=:circle, markersize=2)
end 
display(p)

# AMH Sampler 
st1 = st1
# st1 = State_θ(randn(nparams(stm1.pot)), zeros(nparams(stm1.pot)))
AMHoutp1 = MHoutp_θ()    
AMHsampler1 = AdaptiveMHsampler(1e-5, st1, stm1, st1.θ, Σ_initial) ;  
@time Samplers.run!(st1, AMHsampler1, stm1, 10000, AMHoutp1)

AMHsampler1.Σ
eigen(AMHsampler1.Σ).values

Histogram(AMHoutp1)
Trajectory(AMHoutp1)
Summary(AMHoutp1)
Histogram(AMHoutp1; save_fig=true, title="AMH_Hist_0.03_FullFS_Preconditioned")
Trajectory(AMHoutp1; save_fig=true, title="AMH_Traj_0.03_FullFS_Preconditioned")
Summary(AMHoutp1; save_fig=true, title="AMH_Summ_0.03_FullFS_Preconditioned")

st2 = State_θ(reshape(true_θ(), nparams(stm1.pot)), zeros(nparams(stm1.pot)))
AMHoutp2 = MHoutp_θ()    
AMHsampler2 = AdaptiveMHsampler(0.1, st2, stm1, st2.θ, get_precon(stm1.pot, .5, 2.0)) ;  
Samplers.run!(st2, AMHsampler2, stm1, 300000, AMHoutp2)
Histogram(AMHoutp2)
Trajectory(AMHoutp2)
Summary(AMHoutp2)
Histogram(AMHoutp2; save_fig=true, title="AMH_Hist_0.1_FullFS_Preconditioned")
Trajectory(AMHoutp2; save_fig=true, title="AMH_Traj_0.1_FullFS_Preconditioned")
Summary(AMHoutp2; save_fig=true, title="AMH_Summ_0.1_FullFS_Preconditioned")

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