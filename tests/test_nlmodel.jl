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

println(fieldnames(typeof(at)))     # (:X, :P, :M, :Z, :cell, :pbc, :calc, :dofmgr, :data)
#                                      X: Positions, P: Momenta, Z: Species, M: Mass(?), 
#                                      pbc: periodic boundary conditions
println(length(at.X))               # 54 points
x_vals = [point[1] for point in at.X]
y_vals = [point[2] for point in at.X] 
z_vals = [point[3] for point in at.X] 
scatter(x_vals, y_vals, z_vals)     # Quick inspection of initial positions of particles 
rattle!(at,0.1)                     # Perturbs the system 
println(at.X[1])

#Create Finnis-Sinclair model, with struct defined in nlmodels.jl
# Type Hierarchy: Any -> FSModel
model = FSModel(basis1, basis2, 
                            rcut, 
                            x -> -sqrt(x), 
                            x -> 1 / (2 * sqrt(x)), 
                            ones(length(basis1)), ones(length(basis2)))
println(fieldnames(FSModel))        # (:basis1, :basis2, :rcut, :transform, :transform_d, :c1, :c2)


#Evaluate potential energy of model at bulk configuration
E = ACE.evaluate(model, at)
using BayesianMLIP.NLModels: evaluate_param_d


nlist = neighbourlist(at, model.rcut) #a neighb
lin_part, nonlin_part = 0.0,0.0
k=1
_, Rs = NeighbourLists.neigs(nlist, k)
cfg = [ ACE.State(rr = r)  for r in Rs ] |> ACEConfig
B1 = evaluate(model.basis1, cfg)



grad1, grad2 = evaluate_param_d(model, at)

grad = cat(grad1,grad2, dims=1)
