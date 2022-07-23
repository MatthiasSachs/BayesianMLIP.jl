using ACEflux, JuLIP, ACE, Flux, Zygote
using ACE: State
using LinearAlgebra
using Printf
using Test, ACE.Testing
using ACEbase

using ACEflux: Linear_ACE
using StaticArrays
using JLD2
#model = Chain(Linear_ACE(2, 7, 4), Dense(4, 3, σ), Dense(3, 1), sum)
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

model = Chain(Linear_ACE(;ord = 2, maxdeg = 4), GenLayer(FS), sum)
pot = FluxPotential(model, 6.0) 
##

# # we only check the derivatives of the parameters in the linear ace layer
# # we do finite difference on the whole function, but only compare ACE parameters

@info "dEnergy, dE/dP"

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 


theta = load_object("Data/dataset1.jld2")["theta"];
data = load_object("Data/dataset1.jld2")["data"];

w0 = 1.0 
weight_E, weight_F = w0, w0/ (3*length(at)) # weights correspond to precision of noise
log_likelihood = (model, d) -> -weight_E * abs(d.E - energy(model,d.at)) -  weight_F * sum(sum(abs2, g - f.rr) 
                     for (g, f) in zip(forces(model, d.at), d.F))
#log_likelihood_Energy = (model, d) -> -1.0 * (d.E - energy(model,d.at))^2


# Much of the computing power is used to solve for the force part of log_likelihood

using Distributions
using BayesianMLIP.Utils: StatisticalModel, get_params

priorNormal = MvNormal(zeros(length(get_params(model))),I)

sm = StatisticalModel(
    log_likelihood, 
    priorNormal, 
    pot, 
    data
);

using BayesianMLIP.Utils: get_glp, get_glpr, get_gll
using BayesianMLIP.NLModels: get_params, nparams
gll = get_gll(sm)
glpr = get_glpr(sm)
glp = get_glp(sm)


c =rand(length(get_params(pot)))

gll(c,data[2])
n = 50
g = glp(c,data)

