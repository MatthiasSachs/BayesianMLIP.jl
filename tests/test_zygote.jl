using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, PIBasis, O3, State, val 
using JuLIP
import JuLIP: forces, energy
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
nX = 10     # number of atoms
cfg = ACEConfig([State(rr = rand(SVector{3, Float64})) for _ in 1:nX])      # vector of States

#initialize the model
np = 1
c_m = rand(SVector{np,Float64}, length(basis))
model = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

# Question: Basis length is 97. Why 97×2 matrix of coefficients c? 

props = evaluate(model, cfg)        # What does this function do? 


##
θ = randn(np * length(basis)) ./ (1:(np*length(basis))).^2  # Rand vector, rescales it 
c = reinterpret(SVector{np, Float64}, θ)                     # Reshapes θ
ACE.set_params!(model, c)                                   # Sets new coefficients 'c' for model


# FS = props -> sum([ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
FS = props -> sum( (1 .+ val.(props).^2).^0.5 ) 

# Question: FS is being summed over 2 props 

fsmodel = cfg -> FS(evaluate(model, cfg))

# @info("check the model and gradient evaluate ok")
fsmodel(cfg)
g = Zygote.gradient(fsmodel, cfg)[1]

struct FSModel
    model1::ACE.LinearACEModel
    model2::ACE.LinearACEModel
    transform
    rcut
end

FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel) = FSModel(model1::ACE.LinearACEModel, model2::ACE.LinearACEModel, x -> sum( (1 .+ val.(x).^2).^0.5 ), 5.0)


function energy(m::FSModel, at::Atoms; nlist = nothing) 
    if nlist === nothing
        nlist = neighbourlist(at, m.rcut)
    end
    lin_part, nonlin_part = 0.0, 0.0
    for i = 1:length(at)    # sum over atoms 
        _, Rs = NeighbourLists.neigs(nlist, i)  
        cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
        lin_part += evaluate(m.model1, cfg)[1].val # sum over basis 
        nonlin_part += m.transform(evaluate(m.model2, cfg))[1] # sum over basis 
    end
    return lin_part + nonlin_part
end

function allocate_F(n::Int)
    return zeros(ACE.SVector{3, Float64}, n)
end

function forces(m::FSModel, at::Atoms; nlist = nothing) 
    if nlist === nothing
        nlist = neighbourlist(at, m.rcut)
    end
    F = allocate_F(length(at))
    fsmodel = cfg -> m.transform(evaluate(m.model2, cfg))   
    for i = 1:length(at)
        Js, Rs = NeighbourLists.neigs(nlist, i)    # Js = indices, Rs = PositionVectors 
        cfg = ACEConfig( [ ACE.State(rr = r)  for r in Rs ] )
        for index in 1:length(Js) 
            F[Js[index]] += -ACE.grad_config(m.model1, cfg)[index][1].rr      # Linear Part, adding Dstate to 3Vector
            F[Js[index]] += -Zygote.gradient(fsmodel, cfg)[1][index].rr     # Nonlinear Part, 
        end  
    end
    return F 
end

# But all force vectors are 0 
forces(m, at) 

nparams(m::FSModel) = nparams(m.model1) + nparams(m.model2)

params(m::FSModel) = copy(m.c)

function set_params!(m::FSModel, c) 
   m.c[:] .= c
   set_params!(m.evaluator, m.basis, c)
   return m 
end

set_params!(::FSModel, args...) = nothing 

