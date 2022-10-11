module Outputschedulers
using ACE 
using JuLIP
using BayesianMLIP.NLModels
export simpleoutp, atoutp, outputscheduler, feed!, MHoutp
export outputscheduler, MHoutp_θ, BAOABoutp_θ, BADODABoutp_θ, atoutp, SGLDoutp_θ 
using ACE: val
abstract type outputscheduler end

mutable struct atoutp <: outputscheduler 
    θ 
end 
atoutp() = atoutp([])

mutable struct MHoutp_θ <: outputscheduler 
    θ
    log_posterior 
    acceptance_rate 
    eigen_ratio
    covariance_metric 
    acceptance_rate_lin
    acceptance_rate_nlin
end 
MHoutp_θ() = MHoutp_θ([], [], [], [], [], [], []) 

mutable struct BAOABoutp_θ <: outputscheduler
    θ 
    θ_prime 
    log_posterior
end 
BAOABoutp_θ() = BAOABoutp_θ([], [], []) 

mutable struct BADODABoutp_θ <: outputscheduler
    θ 
    θ_prime 
    log_posterior
    ξ
end 
BADODABoutp_θ() = BADODABoutp_θ([], [], [], []) 

mutable struct SGLDoutp_θ <: outputscheduler 
    θ 
    log_posterior 
end 
SGLDoutp_θ() = SGLDoutp_θ([], [])

end # end module 

