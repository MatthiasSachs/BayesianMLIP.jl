using ACE, ACEatoms, Plots, ACEflux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Distributions
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics  
using BayesianMLIP.MiniACEflux, BayesianMLIP.Utils, BayesianMLIP.Samplers, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser


# Initialize Finnis-Sinclair Model with ACE basis (w/ coefficients=0)
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 2), GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 6.0); 

# Can call nparams to find number of params, No problems here 
nparams(pot)

# Can call get_params to find parameters of pot, but this should be a 2×15 matrix rather than a 30-vector 
get_params(pot) 

# Can't call set_params! to set parameters of pot. 
# This should take in pot as the first argument and a 30-vector as the second argument. 
# set_params!(pot, [1, 2, ..., 30]) should set the parameter models to be matrix [1 3 5 ... 29 ; 2 4 6 ... 30]  
set_params!(pot, randn(30)) 