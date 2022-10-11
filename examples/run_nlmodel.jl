using ACE, ACEatoms, ACEflux, Flux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics, JSON
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics
using BayesianMLIP.Utils, BayesianMLIP.Samplers, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser
import ACEflux: FluxPotential
import Distributions: MvNormal

at = bulk(:Cu, cubic=true) * 3; rattle!(at, 0.1) ; 

# Initialize Finnis-Sinclair Model 
rcut = 3.0
# FS is the transformation of the basis evaluated at each atomic environment, then you sum everything up 
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10; 
model = Chain(Linear_ACE(;ord = 2, maxdeg = 1, Nprop = 2, rcut=rcut), GenLayer(FS));
pot = ACEflux.FluxPotential(model, rcut); 

function log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = ωE/(3*length(d.at)) )
    # Compute the log_likelihood for one data point: log P(θ|d)
    -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) 
                     for (g, f) in zip(forces(pot, d.at), d.F))
end 

log_likelihood_Null = ConstantLikelihood() 

# Generate positive definite matrix 
priorNormal = MvNormal(zeros(nparams(pot)), I)
priorUniform = FlatPrior()

# real data 
real_data = getData(JSON.parsefile("/z1-mbahng/mbahng/mlearn/data/Cu/training.json"))[1:3] ; # 262-vector 


stm1 = StatisticalModel(log_likelihood_Null, priorNormal, pot, real_data) ;
stm1 = StatisticalModel(log_likelihood_L2, priorUniform, pot, real_data) ;

hyperparams = precon_pre_cov_mean(stm1)
true_closed_mu_lin = hyperparams["true_mean"]
true_closed_pre_lin = hyperparams["true_precision"]
true_closed_cov_lin = hyperparams["true_covariance"]

singDec = svd(true_closed_pre_lin)
true_closed_pre_nlin = Symmetric(singDec.U * Diagonal(singDec.S .^2) * transpose(singDec.U))

m = nlinparams(stm1)

# Run Gibbs sampler on conditional distribution of linear components 
init_mu = vcat(true_closed_mu_lin, zeros(m)) 

st1 = State_θ(init_mu, zeros(nparams(pot))) ; 
AMHoutp1 = MHoutp_θ() ; 
Gibbsampler1 = GibbsSampler(1e1, 1e4, st1, stm1, true_closed_mu_lin, true_closed_mu_lin, true_closed_pre_lin, true_closed_pre_nlin, 0.9, 0.99); 
Samplers.run!(st1, Gibbsampler1, stm1, 30000, AMHoutp1, 0.) 

Summary(AMHoutp1)
Histogram(AMHoutp1)
Trajectory(AMHoutp1) 
Histogram(AMHoutp1; save_fig=true, title="GibbsNonLinHist2")
Trajectory(AMHoutp1; save_fig=true, title="GibbsNonLinTraj2")
Summary(AMHoutp1; save_fig=true, title="GibbsNonLinSumm2")

dict = Dict{String, Any}("st1" => st1, 
                         "AMHsampler1"  => AMHsampler1, 
                         "stm1" => stm1, 
                         "AMHoutp1" => AMHoutp1)
save("continue_running.jld2", dict)