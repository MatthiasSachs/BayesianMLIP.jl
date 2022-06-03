module Outputschedulers 

export feed!
export simpleoutp, outputscheduler, feed!

abstract type outputscheduler end

struct simpleoutp <: outputscheduler
    X_traj
    P_traj 
end

simpleoutp() = simpleoutp([], [])

function feed!(d, V, at, outp::simpleoutp)
    push!(outp.X_traj, copy(at.X))
    push!(outp.P_traj, copy(at.P))
end


end # end module 