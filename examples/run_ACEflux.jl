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


theta = load_object("Data/dataset1.jld2")["theta"];
data = load_object("Data/dataset1.jld2")["data"];

w0 = 1.0 
weight_E, weight_F = w0, w0/ (3*length(at)) # weights correspond to precision of noise
log_likelihood = (model, d) -> -weight_E * (d.E - energy(model,d.at))^2 -  weight_F * sum(sum(abs2, g - f.rr) 
                     for (g, f) in zip(forces(model, d.at), d.F))


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
gll = get_gll(sm) # gradient of log-likelihood over a single date point
glpr = get_glpr(sm) # gradient of the log-prior
glp = get_glp(sm) # gradient of the log-posterior over data set

c = zeros(Flux.params(sm.model))
c =rand(length(c))

data[1].E
gll(c,data[1])


glp(c,data[1:2])
n = 10

g_estimate = glp(c,data[1:n])/n

glp(c,data)





