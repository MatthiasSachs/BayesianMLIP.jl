module Outputschedulers
using ACE 
using JuLIP
using BayesianMLIP.NLModels
export simpleoutp, atoutp, outputscheduler, feed!, MHoutp
using ACE: val
abstract type outputscheduler end

struct atoutp <: outputscheduler 
    at_traj
end 
atoutp() = atoutp([])



end # end module 

