using ACE
using ACEatoms 
using BayesianMLIP           
using BayesianMLIP.NLModels         
using BayesianMLIP.Dynamics
using BayesianMLIP.Outputschedulers
using BayesianMLIP.Utils
using Random: seed!, rand
using JuLIP
using Plots 
using ACE: O3, evaluate
using StaticArrays
using ACE: val, SymmetricBasis
using BayesianMLIP.NLModels: NLModel, CombPotential, svecs2vec, pack, unpack
using BayesianMLIP.ACEflux
using BayesianMLIP.ACEflux: FluxPotential

using Test

function createFSmodel(maxdeg, ord)
    # construct two bases
    Bsel = SimpleSparseBasis(ord, maxdeg)
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
    φ = ACE.Invariant()
    basis1 = SymmetricBasis(φ, B1p, O3(), Bsel)

    Bsel = SimpleSparseBasis(ord, maxdeg)
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
    φ = ACE.Invariant()
    basis2 = SymmetricBasis(φ, B1p, O3(), Bsel)

    #Define generalized Finnis-Sinclair model
    FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
    ID = props -> sum( val.(props) )
    np = 1
    c_m1 = rand(SVector{np,Float64}, length(basis1));
    nlm1 = NLModel(ACE.LinearACEModel(basis1, c_m1, evaluator = :standard), np, ID);
    c_m2 = rand(SVector{np,Float64}, length(basis2));
    nlm2 = NLModel(ACE.LinearACEModel(basis2, c_m2, evaluator = :standard), np, FS);

    rcut = 2*rnn(:Al)

    model = CombPotential(FluxPotential(nlm1, rcut),FluxPotential(nlm2, rcut));# this is the FS-model
    return model
end 

model = createFSmodel(6, 2)
c = params(model)

at = bulk(:Al, cubic=true) * 2 
rattle!(at,.5)

using Distributions
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.NLModels: params
using JLD2

sampler = BAOAB(0.001, model, at)

Ndata = 100
data = load_object("data.jld2")
for k = 1:Ndata
    run!(sampler, model, at, 1000; outp=nothing)
    push!(data,(at= deepcopy(at), E = energy(model, at), F= forces(model, at))) 
end

# using JLD2
# save_object("data.jld2", data)
# k = load_object("data.jld2")

# Define log_likelihood function
w0 = 1.0 
weight_E, weight_F = w0, w0/ (3*length(at)) # weights correspond to precision of noise

# Log likelihood (Negative cost) for single data point, defined to be the sum of the squares of the elementwise differences
# Modification: Likelihood function only includes energy observations 
log_likelihood = (model, d) -> -weight_E * (d.E - energy(model,d.at))^2

# Define prior 
prior = MvNormal(zeros(length(c)),I) 

# Define log posterior using formula p(θ | D ) ∝ p(D | θ) p(θ) = p(θ) ∏_i^N p(d_i | θ) ⟹ ...
function log_posterior(model, data, prior, log_likelihood)
    return sum(log_likelihood(model, d) for d in data) + logpdf(prior, c)
end

function U(θ)
    set_params!(model, θ) 
    return -log_posterior(model, data, prior, log_likelihood)
end 

theta = randn(72)
U(theta)

using Zygote
using ForwardDiff
# Need to run BAOAB on potential U 
inp = ones(72)
U(inp)
Zygote.gradient((x...) -> sum([i^2 for i in x]), inp...)
Zygote.gradient(U, inp)



using BayesianMLIP.MHoutputschedulers
using BayesianMLIP.Samplers 

outp = MHoutp()
MetroHastings = SimpleMHsampler(randn(length(c12)), randn(length(c12)), I)


BayesianMLIP.Samplers.run!(MetroHastings, 1000, model, log_posterior, outp)

for step in outp.θ_steps
    println(step[1])
end