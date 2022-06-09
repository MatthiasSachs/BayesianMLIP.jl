using ACE
using ACEatoms                      # ACEatoms depends on ACE, the dependeny error is most 
using BayesianMLIP.NLModels         # likely an incompatibility with some of these packages 
using BayesianMLIP.Dynamics
using BayesianMLIP.Outputschedulers
using BayesianMLIP.Utils
using Random: seed!, rand
using JuLIP
using Plots          

maxdeg = 4 # max degree
ord = 2 # max body order correlation 
Bsel = SimpleSparseBasis(ord, maxdeg)
rcut = 5.0 # Cutoff radius

# Create phi_mnl one particle basis
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, rin = 1.2, rcut = 5.0)

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


rattle!(at,0.1)                     # Perturbs the system, changes the input 'at' 
r0 = rnn(:Al)
model = FSModel(basis1, basis2, rcut, x -> -sqrt(x+0.1), x -> 1 / (2 * sqrt(x+0.1)), rand(length(basis1)), zeros(length(basis2)))
model = JuLIP.morse(;A=4.0, e0=.5, r0=r0, rcut=(1.9*r0, rcut))                    


sampler = BAOAB(0.01, model, at) 
# sampler = VelocityVerlet(0.05, model, at)
outp = atoutp()
outp2 = atoutp()
nsteps = 5000
run!(sampler, model, at, nsteps; outp=outp2)
animation(outp2)
length(outp2.at_traj)

for j in 1:1 # 54 total, particle 33 and 50
    println("------------------------------------")
    for i in 50:50:5000
        println("Step $(i)")
        println("Particle $(j) Position:  ", outp.at_traj[i].X[j])
        println("Particle $(j) Momenta:   ", outp.at_traj[i].P[j])
        println("Particle $(j) Force:     ", outp.forces[i][j])
        println("System Potential Energy: ", outp.energy[i])
        println("System Hamiltonian:      ", outp.Hamiltonian[i])
        println("")
    end 

end 