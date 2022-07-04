module Samplers
using Distributions
using BayesianMLIP.NLModels
using BayesianMLIP.Utils
using BayesianMLIP.MiniACEflux: FluxPotential
using LinearAlgebra
using BayesianMLIP.Outputschedulers

using BayesianMLIP.MHoutputschedulers: MHoutp

export SimpleMHsampler, AdaptiveMHsampler, step!, run!

abstract type sampler end 
abstract type MHsampler end 

mutable struct SimpleMHsampler <: MHsampler 
    θ ::Vector{Float64}
    log_posterior_val::Float64      # value of target density at current param value θ
    step::Int64
    β::Float64          # Inverse temperature
end 
SimpleMHsampler(θ) = SimpleMHsampler(θ, 0.0, 1, 2.0)   #β initialized to 2.0 so that scaling factor is 1/log(2) 

mutable struct AdaptiveMHsampler <: MHsampler
    θ ::Vector{Float64}
    μ ::Vector{Float64}
    Σ
    log_posterior_val::Float64
    step::Int64
    β::Float64          # Inverse temperature
end
AdaptiveMHsampler(θ, μ, Σ) = AdaptiveMHsampler(θ, μ, Σ, 0.0, 1, 2.0) #β initialized to 2.0 so that scaling factor is 1/log(2) 

function set_β!(mhsampler::MHsampler, β::Float64)
    mhsampler.β = β
    return nothing 
end 

function get_β!(mhsampler::MHsampler) 
    return mhsampler.β
end

function step!(mhsampler::SimpleMHsampler, m::StatisticalModel)
    # Update the log posterior value of the model to its dataset (with specified prior) at current parameter θ
    if mhsampler.step == 1
        mhsampler.log_posterior_val = log_posterior(m, mhsampler.θ)
    end 

    # Calculate θ' from proposal distribution
    θ_prime = rand(MvNormal(mhsampler.θ, (1/log(mhsampler.β)*I)))

    # Initialize models with parameters θ_k and θ_prime 
    proposal_log_prob = log_posterior(m, θ_prime)

    logU = log(rand())      # Uniform[0, 1]
    
    if logU < min(0, proposal_log_prob - mhsampler.log_posterior_val)       # Accept 
        mhsampler.θ = θ_prime           # Update to proposed state
        mhsampler.log_posterior_val = log_posterior(m, mhsampler.θ)     # Update log_posterior value at new θ
        set_β!(mhsampler, mhsampler.β + 1.0)
    else logU ≥ min(0, proposal_log_prob - mhsampler.log_posterior_val)     # Reject 
        # mhsampler.θ = mhsampler.θ       # Keep state the same
    end 

    mhsampler.step += 1
end

function step!(mhsampler::AdaptiveMHsampler, m::StatisticalModel)
    # Update the log posterior value of the model to its dataset (with specified prior) at current parameter θ
    if mhsampler.step == 1
        mhsampler.log_posterior_val = log_posterior(m, mhsampler.θ)
    end 

    # Calculate θ' from proposal distribution
    θ_prime = rand(MvNormal(mhsampler.θ, (1/log(mhsampler.β)) * mhsampler.Σ))

    # Initialize models with parameters θ_k and θ_prime 
    proposal_log_prob = log_posterior(m, θ_prime)

    logU = log(rand())      # Uniform[0, 1]
    
    if logU < min(0, proposal_log_prob - mhsampler.log_posterior_val) 
        mhsampler.θ = θ_prime           # Update to proposed state
        mhsampler.log_posterior_val = log_posterior(m, mhsampler.θ)     # Update log_posterior value at new θ
        set_β!(mhsampler, mhsampler.β+1)

        # Update mean and covariance 
        # Mean: μ_{k+1} = μ_k + \frac{1}{k+1} (θ_{k+1} - μ_k) 
        # Covariance: Σ_{k+1} = Σ_k + \frac{1}{k+1} [(θ_{k+1} - μ_k)(θ_{k+1} - μ_k)^T - Σ_k]
        mhsampler.Σ = mhsampler.Σ + (1/(mhsampler.step +1))*(((mhsampler.θ - mhsampler.μ) * transpose(mhsampler.θ - mhsampler.μ)) - mhsampler.Σ)
        mhsampler.μ = mhsampler.μ + (1/(mhsampler.step +1))*(mhsampler.θ - mhsampler.μ)     # Must be updated second

    else logU ≥ min(0, proposal_log_prob - mhsampler.log_posterior_val)
        # mhsampler.θ = mhsampler.θ       # Keep state the same
    end 

    mhsampler.step += 1
end

function run!(sampler::MHsampler, stat_model, nsteps::Int64, outp::MHoutp)
    push!(outp.θ_steps, sampler.θ)          # Should implement feed function? 
    for i in 1:nsteps 
        step!(sampler, stat_model);
        push!(outp.θ_steps, sampler.θ)
        push!(outp.log_posterior, sampler.log_posterior_val)
        # distance = norm(Theta - sampler.θ)
        # push!(outp.metric, distance)
        println(i, " : ", sampler.log_posterior_val)
    end 
end 


end