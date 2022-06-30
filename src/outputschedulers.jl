module Outputschedulers
using ACE 
using JuLIP
using BayesianMLIP.NLModels
export simpleoutp, atoutp, outputscheduler, feed!, MHoutp
using ACE: val
abstract type outputscheduler end

struct atoutp <: outputscheduler 
    at_traj
    energy
    kenergy
    forces
    hamiltonian
end 
atoutp() = atoutp([], [], [], [], [])

function feed!(V, at, outp::atoutp)

    E = energy(V, at)
    KE = kenergy(at)

    push!(outp.at_traj, deepcopy(at))
    push!(outp.energy, E)
    push!(outp.kenergy, KE)
    push!(outp.forces, forces(V, at))
    push!(outp.hamiltonian, E+KE)
    #@show hamiltonian(V,at)
    #@show energy(V, at)
end


end # end module 

