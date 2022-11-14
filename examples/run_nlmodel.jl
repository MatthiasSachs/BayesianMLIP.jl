using ACE, ACEatoms, ACEflux, Flux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics, JSON, Plots, BenchmarkTools
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

st = State_θ(vcat([-91.23915391, -66.78643017, 66.45443175], zeros(3)), randn(6)) 
output = outp() ; 
s = linearSGLD(1e-13, st, stm, 1; β=Inf, α=0.9) ;
conditionalSamplers.run!(st, s, stm, 20000, output) 
delete_first!(output, 100)
delete_last!(output, 10) 
s.Σ
Summary(output)
Histogram(output)
Trajectory(output) 

scatter(
    [x[1] for x in output.θ], 
    [x[2] for x in output.θ], 
    legend=false, 
    markersize=1, 
)


st1 = State_θ(vcat([-91.23915391, -66.78643017, 66.45443175], randn(3))) 
output1 = outp() ; 
s1 = nonlinearSGLD(1e-14, st1, stm, 1; β=1e-9, α=1.0) ;
conditionalSamplers.run!(st1, s1, stm, 10000, output1) 
delete_first!(output1, 1000)

Summary(output1)
Histogram(output1)
Trajectory(output1) 




D = svd(Sig); STD = D.U * Diagonal(sqrt.(D.S)) * transpose(D.U)

x = ones(6)
lp1 = get_lp(stm, Mu, STD);
lp2 = get_lp(stm, zeros(6), Diagonal(ones(6)));
glp1 = get_glp(stm, Mu, STD) ;
glp2 = get_glp(stm, zeros(6), Diagonal(ones(6))) ;

lp1(x, stm.data, length(stm.data))==lp2(STD * x + Mu, stm.data, length(stm.data))

glp1(x, stm.data, length(stm.data))
STD * (glp2(STD * x + Mu, stm.data, length(stm.data)))

