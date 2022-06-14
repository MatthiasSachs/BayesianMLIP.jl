using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, PIBasis, O3, State, val 
using StaticArrays
using ChainRules
import ChainRulesCore: rrule, NoTangent, ZeroTangent
using Zygote
using Zygote: @thunk 
using Printf, LinearAlgebra #for the fdtestMatrix

##

@info("differentiable model test")

# construct the basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

# generate a random configuration
nX = 10
cfg = ACEConfig([State(rr = rand(SVector{3, Float64})) for _ in 1:nX])

#initialize the model
np = 2
c_m = rand(SVector{np,Float64}, length(basis))
model = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

evaluate(model, cfg)

##

θ = randn(np * length(basis)) ./ (1:(np*length(basis))).^2
c = reinterpret(SVector{2, Float64}, θ)
ACE.set_params!(model, c)

# FS = props -> sum([ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
FS = props -> sum( (1 .+ val.(props).^2).^0.5 )
fsmodel = cfg -> FS(evaluate(model, cfg))

# @info("check the model and gradient evaluate ok")
fsmodel(cfg)
g = Zygote.gradient(fsmodel, cfg)[1]

# begin struct FSModel
#     model1::ACE.LinearACEModel
#     model2::ACE.LinearACEModel
#     transform
#     rcut
# end

# FSModel(model1::ACE.LinearACEModel,model2::ACE.LinearACEModel ; transform= x -> sum( (1 .+ val.(x).^2).^0.5 )) = FSModel(model1::ACE.LinearACEModel,model2::ACE.LinearACEModel, transform)

# function energy(m::FSModel, at::Atoms; nlist = nothing) 
    
# end
# function forces(m::FSModel, at::Atoms; nlist = nothing) 
#     if nlist === nothing
#         nlist = neighbourlist(at, m.rcut)
#     end
#     fsmodel = cfg -> m.transform(evaluate(m.model2, cfg))
#     for k = 1:length(at)
#         Js, Rs = NeighbourLists.neigs(nlist, k)
#         cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
#         F[Js] += -ACE.grad_config(m.model1, cfg) 
#         F[Js] += -Zygote.gradient(fsmodel, cfg)[1]
#     end
#     return F 
# end

# nparams(m::FSModel) = nparams(m.model1) + nparams(m.model2)

# params(m::FSModel) = copy(m.c)

# function set_params!(m::FSModel, c) 
#    m.c[:] .= c
#    set_params!(m.evaluator, m.basis, c)
#    return m 
# end

# set_params!(::FSModel, args...) = nothing 

