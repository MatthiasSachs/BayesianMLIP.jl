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

function log_likelihood_L1(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = 1.0/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * abs(d.E - energy(pot, d.at)) -  ωF * sum(sum(abs2, g - f) 
                     for (g, f) in zip(forces(pot, d.at), d.F))
end 

priorNormal = MvNormal(zeros(length(get_params(pot))),I)

Data = getData(JSON.parsefile("./Run_Data/Real_Data/training_test/Cu/training.json")) # 262-vector 


# Initialize StatisticalModel 
stm1 = StatisticalModel(log_likelihood_L1, priorNormal, pot, Data); 
log_likelihood(stm1)
log_prior(stm1)
log_posterior(stm1)





function saveData() 
    Threads.@threads for i in 4:9 
        st = State_θ(rand(30), zeros(30))
        AMHoutp = MHoutp_θ()    # new run 
        AMHsampler = AdaptiveMHsampler(0.01, st, stm1, st.θ, I)     
        Samplers.run!(st, AMHsampler, stm1, 1000, AMHoutp)

        # Save last state, last state of sampler, statistical model, and outp to jld2 file 
        dict = Dict{String, Any}( "description" => :"1000 steps run on Cu Training Data on Adaptive Metropolis Hastings", 
                                "state" => st, 
                                "sampler" => AMHsampler, 
                                "stm" => stm1, 
                                "outp" => AMHoutp)

        save("./Run_Data/AMH_Cu_Training$i/AMH_Cu_Training$i-1000.jld2", dict)
    end 
end 

info = load("./Run_Data/artificial_data.jld2")
info["theta"]
info["data"]

st = State_θ(rand(30), zeros(30))
AMHoutp = MHoutp_θ()    # new run 
AMHsampler = AdaptiveMHsampler(0.01, st, stm1, st.θ, I)     
Samplers.run!(st, AMHsampler, stm1, 1000, AMHoutp)

# Save last state, last state of sampler, statistical model, and outp to jld2 file 
dict = Dict{String, Any}( "description" => :"1000 steps run on Cu Training Data on Adaptive Metropolis Hastings", 
                          "state" => st, 
                          "sampler" => AMHsampler, 
                          "stm" => stm1, 
                          "outp" => AMHoutp)

save("./Run_Data/AMH_Cu_Training3/AMH_Cu_Training3-1000.jld2", dict) 

info1 = load("./Run_Data/AMH_Cu_Training1/AMH_Cu_Training1-1081.jld2")
outp1 = info1["outp"]

plotLogPosterior(outp1)

info2 = load("./Run_Data/AMH_Cu_Training2/AMH_Cu_Training2-1000.jld2")




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