module Outputschedulers
using ACE 
using BayesianMLIP.NLModels
export simpleoutp, atoutp, outputscheduler, feed!

abstract type outputscheduler end

struct simpleoutp <: outputscheduler
    X_traj
    P_traj 
end
simpleoutp() = simpleoutp([], [])

struct atoutp <: outputscheduler 
    at_traj
    energy
    forces
end 
atoutp() = atoutp([], [], [])

function feed!(V, at, outp::simpleoutp)
    push!(outp.X_traj, deepcopy(at.X))
    push!(outp.P_traj, deepcopy(at.P))
end

function feed!(V, at, outp::atoutp)
    push!(outp.at_traj, deepcopy(at))
    push!(outp.energy, energy(V, at))
    push!(outp.forces, forces(V, at))
end


end # end module 