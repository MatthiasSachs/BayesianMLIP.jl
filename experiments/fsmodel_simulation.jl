using BayesianMLIP.NLModels
using BayesianMLIP.Dynamics
using ACE, JuLIP
 

maxdeg = 4
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)
rcut = 5.0 

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
                                    rin = 1.2, rcut = 5.0)
ACE.init1pspec!(B1p, Bsel)
basis1 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)


at = bulk(:Ti, cubic=true) * 3

rattle!(at,0.1) 
model = FSModel(basis1, basis2, 
                    rcut, 
                    x -> -sqrt(x+0.1), 
                    x -> 1 / (2 * sqrt(x + 0.1)), 
                    ones(length(basis1)), zeros(length(basis2)))

using BayesianMLIP.Dynamics
using Copy
at = bulk(:Ti, cubic=true) * 3
N_obs =1000
nsteps = 1000
sampler = BAOAB(0.01, model, at; γ=1.0, β=1.0)
data = []
for k in 1:N_obs
    run!(sampler, model, at, nsteps; outp = nothing)
    push!(data, (at = deepcopy(at), E= energy(model,at), F = forces(model,at) ))
end

struct GaussianNoiseLL
    model
    variance
    data
end

function logLikelihood(sm::GaussianNoiseLL, c1, c2 )

end
function logLikelihood_d(sm::GaussianNoiseLL, c1, c2)

end
# 1) Check that force works properly for linear version of fsmodel  
# 2) Check regularized non-linear version ( + 0.1, sigmoid?, arctan?)
# 3) Create synthetic data from a linear model with order = 3 or higher
# 4) Use non-linear FSmodel with order = 2 to learn same potential       
# 5) 