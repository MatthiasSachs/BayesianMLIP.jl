using ACE, ACEatoms, ACEflux, Flux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics, JSON
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Utils, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser, BayesianMLIP.globalSamplers, BayesianMLIP.conditionalSamplers
import ACEflux: FluxPotential 
import Distributions: MvNormal

rcut = 3.0; FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10; 
model = Chain(Linear_ACE(;ord = 2, maxdeg = 1, Nprop = 2, rcut=rcut), GenLayer(FS));
pot = ACEflux.FluxPotential(model, rcut); 

log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = ωE/(3*length(d.at)) ) = -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) for (g, f) in zip(forces(pot, d.at), d.F))
log_likelihood_Null = ConstantLikelihood() 
priorNormal = MvNormal(zeros(nparams(pot)), I)
priorUniform = FlatPrior()
real_data = getData(JSON.parsefile("./Run_Data/Real_Data/training_test/Cu/training.json"))[1:5] ; # 262-vector 


# stm1 = StatisticalModel(log_likelihood_Null, priorNormal, pot, real_data) ;
stm = StatisticalModel(log_likelihood_L2, priorUniform, pot, real_data) ;

# Run MCMC to get to minimum, and then run SGLD (linear parameters) 
st1 = State_θ(zeros(6)) ; 
outp1 = MHoutp_θ() ; 
s1 = linearMetropolis(1e1, st1, stm; α=0.9);
conditionalSamplers.run!(st1, s1, stm, 10000, outp1)
Summary(outp1)
Histogram(outp1)
Trajectory(outp1) 
mean(outp1.θ)


st2 = State_θ(mean(outp1.θ)) ;
outp2 = MHoutp_θ() ; 
s2 = nonlinearSGLD(4e-8, st2, stm, 1; α=0.9); 
conditionalSamplers.run!(st2, s2, stm, 6000, outp2)
Summary(outp2)
Histogram(outp2)
Trajectory(outp2) 

# Run MCMC to get to minimum, and then run SGLD (nonlinear parameters) 
st3 = State_θ(zeros(6)) ; 
outp3 = MHoutp_θ() ; 
s3 = nonlinearMetropolis(1e1, st3, stm; α=0.9);
conditionalSamplers.run!(st3, s3, stm, 10000, outp3)

st4 = State_θ(mean(outp3.θ)) ;
outp4 = MHoutp_θ() ; 
s4 = nonlinearSGLD(1e-9, st4, stm, 1; α=0.9); 
conditionalSamplers.run!(st4, s4, stm, 6000, outp4)
Summary(outp4)
Histogram(outp4)
Trajectory(outp4) 


st5 = State_θ(zeros(nparams(stm))) ; 
outp5 = MHoutp_θ() ; 
s5 = GibbsSampler(
    linearMetropolis(1e1, st5, stm; α=0.9), 
    nonlinearMetropolis(1e1, st5, stm; α=0.6), 
    0.5
);
conditionalSamplers.run!(st5, s5, stm, 50000, outp5)
Summary(outp5)
Histogram(outp5)
Trajectory(outp5)


st6 = State_θ(mean(outp5.θ)) 
outp6 = MHoutp_θ() ; 
s6 = GibbsSampler(
    linearMetropolis(1e1, st6, stm; α=0.9), 
    nonlinearSGLD(1e-7, st6, stm, 1; α=0.9), 
    .5
);
conditionalSamplers.run!(st6, s6, stm, 50000, outp6)

Summary(outp6)
Histogram(outp6)
Trajectory(outp6)
Summary(outp6, save_fig=true, title="summary")
Histogram(outp6, save_fig=true, title="histogram")
Trajectory(outp6, save_fig=true, title="trajectory") 