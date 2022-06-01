using Revise
Revise.includet("../src/nlmodels.jl")

using ACE
using ACEatoms
using BayesianMLIP.NLModels
using Random: seed!, rand
using JuLIP
using Plots             # Add Plots pkg for data visualization 

maxdeg = 4 # max degree
ord = 2 # max body order correlation 
Bsel = SimpleSparseBasis(ord, maxdeg)
rcut = 5.0 # Cutoff radius

# Create phi_mnl one particle basis
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
                                 rin = 1.2, rcut = 5.0)

# B1p -- B1p.bases --- 
#     -- B1p.indices : [(1, 1), (1, 2), ... (3, 2), (4, 1), (3, 3), (3, 4)]
#     -- B1p.B_pool -- B1p.B_pool.arrays -- index

# Create Symmetric bases for fmmodel
ACE.init1pspec!(B1p, Bsel)
basis1 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)

# Create bulk configuration
at = bulk(:Ti, cubic=true) * 3      
# Quick Inspection of at object 
# Type Hierarchy: AbstractVector{StaticArrays.SVector{3, Float64}} 
#                 -> DenseVector{StaticArrays.SVector{3, Float64}} 
#                   -> Vector{StaticArrays.SVector{3, Float64}} * 
# Type Vector with elements of Type StaticArrays.SVector{3, Float64} 

# println(fieldnames(typeof(at)))     # (:X, :P, :M, :Z, :cell, :pbc, :calc, :dofmgr, :data)
#                                      X: Positions, P: Momenta, Z: Species, M: Mass(?), 
#                                      pbc: periodic boundary conditions


# Restructure into list of 3 elements: 54-vector of X-values, Y-values, Z-values
# XYZ_Coords = [ [point[1] for point in at.X], [point[2] for point in at.X], [point[3] for point in at.X] ]
# scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3])     # Quick inspection of initial positions of particles 

rattle!(at,0.1)                     # Perturbs the system, changes the input 'at' 


#Create Finnis-Sinclair model, with struct defined in nlmodels.jl
model = FSModel(basis1, basis2, 
                    rcut, 
                    x -> -sqrt(x), 
                    x -> 1 / (2 * sqrt(x)), 
                    ones(length(basis1)), ones(length(basis2)))

using BayesianMLIP.NLModels: eval_Model, eval_param_gradients, forces 

#Evaluate potential energy of model at bulk configuration
E = eval_Model(model, at)         #TASK: Write code that simulates this system using Langevin dynamics

# Evaluate gradient w.r.t. position vectors of particles 
F = forces(model, at)

# Evaluate the gradients w.r.t. linear parameters c1 and c2
grad1, grad2 = eval_param_gradients(model, at)
println(grad1)
println(grad2)
grad = cat(grad1, grad2, dims=1)       # Concatenate the two arrays
