module NLModels
using ACE, ACEatoms, Flux, JuLIP, ACEflux, Zygote, StaticArrays, NeighbourLists
import ACEflux: FluxPotential
using ACE: evaluate, val, AbstractConfiguration
import JuLIP: forces, energy
using LinearAlgebra: dot

import ACE: set_params!, nparams, params, evaluate, LinearACEModel, AbstractACEModel
export get_params, nparams, set_params!, nlinparams
export energy, forces, site_energy, site_forces, basis_energy, basis_forces

function nparams(pot::FluxPotential) 
    return length(pot.model.layers[1].weight)
end

function nlinparams(pot::FluxPotential) 
    return Int(length(pot.model.layers[1].weight)/2)
end

function get_params(pot::FluxPotential) 
    # returns vector of linear parameters first, then nonlinear 
    return Vector{Float64}(reshape(transpose(pot.model.layers[1].weight), nparams(pot)))
end 

function set_params!(pot::FluxPotential, params::Vector{Float64}) 
    if length(params) != nparams(pot) 
        throw(error("Length $(length(params)) does not match model number of parameters $(nparams(pot))"))
    end 
    matrix = transpose(reshape(params, size(transpose(pot.model.layers[1].weight))))
    pot.model.layers[1].weight = matrix
    ACE.set_params!(pot.model.layers[1].m, ACEflux.matrix2svector(matrix))
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
    no_lin_copies = size(pot.model.layers[1].weight)[1]
    cCu = get_params(pot)[1:Int(nparams(pot)/no_lin_copies)]
    models = Dict(:Cu => ACE.LinearACEModel(basis, cCu; evaluator = :standard))
    V = ACEatoms.ACESitePotential(models)
    V_basis = ACEatoms.basis(V)
    energy(V_basis, at)
end 

function basis_forces(pot::FluxPotential, at::AbstractAtoms)
    # Gets all the basis evaluation of entire configuration 
    basis = pot.model.layers[1].m.basis
    no_lin_copies = size(pot.model.layers[1].weight)[1]
    cCu = get_params(pot)[1:Int(nparams(pot)/no_lin_copies)]
    models = Dict(:Cu => ACE.LinearACEModel(basis, cCu; evaluator = :standard))
    V = ACEatoms.ACESitePotential(models)
    V_basis = ACEatoms.basis(V)
    forces(V_basis, at)
end 

mat2svecs(M::AbstractArray{T}, nc::Int) where {T} =   
      collect(reinterpret(SVector{nc, T}, M))
svecs2vec(M::AbstractVector{<: SVector{N, T}}) where {N, T} = 
      collect(reinterpret(T, M))

end
