module Outputschedulers
export atoutp, outp, delete_first!, delete_last!


abstract type outputscheduler end

mutable struct atoutp <: outputscheduler 
    θ 
end 
atoutp() = atoutp([])

mutable struct outp <: outputscheduler 
    θ
    log_posterior 
    acceptance_rate 
    eigen_ratio
    covariance_metric 
    acceptance_rate_lin
    acceptance_rate_nlin
    mass 
end 
outp() = outp([], [], [], [], [], [], [], []) 

function delete_first!(otp::outp, first_n::Int64) 
    if length(otp.θ) >= first_n  
        otp.θ = otp.θ[first_n+1:end]
    end 
    if length(otp.log_posterior) >= first_n  
        otp.log_posterior = otp.log_posterior[first_n+1:end]
    end 
    if length(otp.acceptance_rate) >= first_n  
        otp.acceptance_rate = otp.acceptance_rate[first_n+1:end]
    end 
    if length(otp.eigen_ratio) >= first_n  
        otp.eigen_ratio = otp.eigen_ratio[first_n+1:end]
    end 
    if length(otp.covariance_metric) >= first_n  
        otp.covariance_metric = otp.covariance_metric[first_n+1:end]
    end 
    if length(otp.acceptance_rate_lin) >= first_n  
        otp.acceptance_rate_lin = otp.acceptance_rate_lin[first_n+1:end]
    end 
    if length(otp.acceptance_rate_nlin) >= first_n  
        otp.acceptance_rate_nlin = otp.acceptance_rate_nlin[first_n+1:end]
    end 
    if length(otp.mass) >= first_n  
        otp.mass = otp.mass[first_n+1:end]
    end 
end 

function delete_last!(otp::outp, first_n::Int64) 
    if length(otp.θ) >= first_n  
        otp.θ = otp.θ[1:end - first_n]
    end 
    if length(otp.log_posterior) >= first_n  
        otp.log_posterior = otp.log_posterior[1:end - first_n]
    end 
    if length(otp.acceptance_rate) >= first_n  
        otp.acceptance_rate = otp.acceptance_rate[1:end - first_n]
    end 
    if length(otp.eigen_ratio) >= first_n  
        otp.eigen_ratio = otp.eigen_ratio[1:end - first_n]
    end 
    if length(otp.covariance_metric) >= first_n  
        otp.covariance_metric = otp.covariance_metric[1:end - first_n]
    end 
    if length(otp.acceptance_rate_lin) >= first_n  
        otp.acceptance_rate_lin = otp.acceptance_rate_lin[1:end - first_n]
    end 
    if length(otp.acceptance_rate_nlin) >= first_n  
        otp.acceptance_rate_nlin = otp.acceptance_rate_nlin[1:end - first_n]
    end 
    if length(otp.mass) >= first_n  
        otp.mass = otp.mass[1:end - first_n]
    end 
end 


end # end module 

