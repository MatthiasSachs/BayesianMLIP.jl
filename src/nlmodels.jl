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

end
