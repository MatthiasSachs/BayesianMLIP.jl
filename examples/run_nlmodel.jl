using ACE, ACEatoms, ACEflux, Flux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics, JSON
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Utils, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser, BayesianMLIP.globalSamplers, BayesianMLIP.conditionalSamplers
import ACEflux: FluxPotential 
import Distributions: MvNormal

rcut = 3.0; FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10; 
model = Chain(Linear_ACE(;ord = 2, maxdeg = 1, Nprop = 2, rcut=rcut), GenLayer(FS));
pot = ACEflux.FluxPotential(model, rcut); 

Mu = randn(6); sig=randn(6, 6); Sig = sig' * sig
log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = ωE/(3*length(d.at)) ) = -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) for (g, f) in zip(forces(pot, d.at), d.F))
log_likelihood_Null = ConstantLikelihood() 
priorNormal = MvNormal(Mu, Sig)
priorUniform = FlatPrior()
real_data = getData(JSON.parsefile("/z1-mbahng/mbahng/mlearn/data/Cu/training.json"))[1:10] ; 
# [-91.23915391, -66.78643017, 66.45443175]
stm = StatisticalModel(log_likelihood_L2, priorUniform, pot, real_data) ;

st = State_θ(vcat([-91.23915391, -66.78643017, 66.45443175], randn(3)), randn(6)) 
output = outp() ; 
s = linearSGLD(4e-11, st, stm, 1; β=1e-5, α=0.9) ;
conditionalSamplers.run!(st, s, stm, 5000, output) 
delete_first!(output, 500)
delete_last!(output, 150)

Summary(output)
Histogram(output)
Trajectory(output) 


st1 = State_θ(vcat([-91.23915391, -66.78643017, 66.45443175], randn(3)), randn(6)) 
output1 = outp() ; 
s1 = nonlinearSGLD2(5e-14, st1, stm, 1; β=1e-10, α=1.0) ;
conditionalSamplers.run!(st1, s1, stm, 10000, output1) 
delete_first!(output1, 300)
delete_last!(output1, 100)

Summary(output1)
Histogram(output1)
Trajectory(output1) 
