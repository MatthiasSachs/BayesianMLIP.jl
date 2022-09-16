module NLModels
using ACE, ACEatoms, Flux, JuLIP, ACEflux
import ACEflux: FluxPotential
using ACE: evaluate, val, AbstractConfiguration
import JuLIP: forces, energy
using NeighbourLists
using Random: seed!
using LinearAlgebra: dot
using Zygote
using StaticArrays

import ACE: set_params!, nparams, params, evaluate, LinearACEModel, AbstractACEModel
export get_params, nparams, set_params!
export energy, forces, site_energy, site_forces, basis_energy, basis_forces
export design_matrix

function get_params(pot::FluxPotential) 
    return pot.model.layers[1].weight
end 

function nparams(pot::FluxPotential) 
    return length(get_params(pot))
end

function set_params!(pot::FluxPotential, params) 
    if prod(size(params)) != prod(size(pot.model.layers[1].weight))
        throw(error("Length does not match parameters of model: $(size(pot.model.layers[1].weight))"))
    end 
    params = reshape(params, size(pot.model.layers[1].weight))
    pot.model.layers[1].weight = params 
    ACE.set_params!(pot.model.layers[1].m, ACEflux.matrix2svector(params))
    return params
end 

function site_energy(pot::FluxPotential, at::AbstractAtoms, i::Int64)
    # energy of ith atomic environment
    pot.model(ACEflux.neighbours_R(pot, at)[i])
end

function site_forces(pot::FluxPotential, at::AbstractAtoms, i::Int64)
    # force of ith atomic environment 
    domain_R = ACEflux.neighbours_R(pot, at)
    J = ACEflux.neighbours_J(pot, at) 
    atomic_env_index = i
    r = domain_R[atomic_env_index]
    tmpfrc = Zygote.gradient(pot, r)[1]
    ith_force = ACEflux.loc_to_glob(tmpfrc, J[atomic_env_index], length(at), atomic_env_index)
    return ith_force
end 

function basis_energy(pot::FluxPotential, at::AbstractAtoms)
    # Gets all the basis evaluation of entire configuration 
    basis = pot.model.layers[1].m.basis
    cCu = reshape(get_params(pot), nparams(pot)) 
    models = Dict(:Cu => ACE.LinearACEModel(basis, cCu; evaluator = :standard))
    V = ACEatoms.ACESitePotential(models)
    V_basis = ACEatoms.basis(V)
    energy(V_basis, at)
end 

function basis_forces(pot::FluxPotential, at::AbstractAtoms)
    # Gets all the basis evaluation of entire configuration 
    basis = pot.model.layers[1].m.basis
    cCu = reshape(get_params(pot), nparams(pot)) 
    models = Dict(:Cu => ACE.LinearACEModel(basis, cCu; evaluator = :standard))
    V = ACEatoms.ACESitePotential(models)
    V_basis = ACEatoms.basis(V)
    forces(V_basis, at)
end 

function design_matrix(pot::FluxPotential, at::AbstractAtoms) 
    # Finds the design matrix 
    bsis_energy = transpose(basis_energy(pot, at))
    bsis_forces = reduce(hcat, svecs2vec.(basis_forces(pot, at)))
    return vcat(bsis_energy, bsis_forces)
end 



mat2svecs(M::AbstractArray{T}, nc::Int) where {T} =   
      collect(reinterpret(SVector{nc, T}, M))
svecs2vec(M::AbstractVector{<: SVector{N, T}}) where {N, T} = 
      collect(reinterpret(T, M))

# struct NLModel <: AbstractACEModel 
#       m::LinearACEModel
#       nc::Int
#       σ
# end

# (nlm::NLModel)(cfg) = evaluate(nlm::NLModel, cfg)
# ACE.evaluate(nlm::NLModel, cfg) = nlm.σ(evaluate(nlm.m, cfg))


# _nc(nlm::NLModel) = nlm.nc

# ACE.params(nlm::NLModel) = svecs2vec(params(nlm.m))
# ACE.nparams(nlm::NLModel) = ACE.nparams(nlm.m)


# ACE.set_params!(nlm::NLModel, c::AbstractArray{T}) where T<:Number = ACE.set_params!(nlm.m, mat2svecs(c, nlm.nc))
# ACE.set_params!(nlm::NLModel, c::AbstractVector{<: SVector{N, T}}) where {N, T}  = ACE.set_params!(nlm.m, c)

# # This provides the standard interface of setter and getter functions to FluxPotentials

# ACE.params(calc::FluxPotential) = params(calc.model)
# ACE.nparams(calc::ACEflux.FluxPotential) = ACE.nparams(calc.model)
# ACE.set_params!(calc::FluxPotential, c) = ACE.set_params!(calc.model, c)
# _nc(calc::FluxPotential) = _nc(calc.model) # This function makes only sense if model is of type NLModel

# # CombPotential provides a workaround to combine two different nonlinear model as a sum of two FluxPotentials
# struct CombPotential <: SitePotential
#     m1::FluxPotential # a function taking atoms and returning a site energy
#     m2::FluxPotential
# end

# function JuLIP.energy(calc::CombPotential, at::Atoms)
#     return energy(calc.m1,at) + energy(calc.m2,at)
# end

# function JuLIP.forces(calc::CombPotential, at::Atoms)
#     return forces(calc.m1,at) + forces(calc.m2,at)
# end

# ACE.nparams(m::CombPotential) = nparams(m.m1) + nparams(m.m2)
# ACE.params(m::CombPotential) = cat(params(m.m1),params(m.m2), dims=1)

    
# function ACE.set_params!(m::CombPotential, c12::AbstractArray{T}) where T<:Number 
#       c1, c2 = unpack(m, c12) 
#       ACE.set_params!(m.m1, c1)
#       ACE.set_params!(m.m2, c2)
#       return m 
# end
# pack(::CombPotential, c1::AbstractArray{T}, c2::AbstractArray{T}) where T<:Number = cat(c1, c2, dims=1)
# pack(::CombPotential, c1::AbstractVector{<: SVector{N, T}}, c2::AbstractVector{<: SVector{N, T}}) where {N, T} = cat(svecs2vec(c1), svecs2vec(c2), dims=1)
# unpack(calc::CombPotential, c12::AbstractArray{T})  where T<:Number = mat2svecs(c12[1:(_nc(calc.m1)*ACE.nparams(calc.m1))],_nc(calc.m1)),  mat2svecs(c12[(_nc(calc.m2)*ACE.nparams(calc.m2)+1):end],_nc(calc.m2))


end
