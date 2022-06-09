module Outputschedulers
using ACE 
using JuLIP
using BayesianMLIP.NLModels
export simpleoutp, atoutp, outputscheduler, feed!

abstract type outputscheduler end

struct atoutp <: outputscheduler 
    at_traj
    energy
    forces
    Hamiltonian
end 
atoutp() = atoutp([], [], [], [])

function feed!(V, at, outp::atoutp)
    push!(outp.at_traj, deepcopy(at))
    push!(outp.energy, energy(V, at))
    push!(outp.forces, forces(V, at))
    push!(outp.Hamiltonian, Hamiltonian(V, at))
end


end # end module 