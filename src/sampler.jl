module Samplers
using Distributions
using BayesianMLIP.MHoutputschedulers
using BayesianMLIP.NLModels
using BayesianMLIP.Utils
using BayesianMLIP.ACEflux: FluxPotential
using LinearAlgebra
using BayesianMLIP.Outputschedulers
export SimpleMHsampler, AdaptiveMHsampler, step!, run!

abstract type sampler end 
abstract type MHsampler end 

mutable struct SimpleMHsampler <: MHsampler 
    θ ::Vector{Float64}
    log_posterior_val::Float64      # value of target density at current param value θ
    step::Int64
end 
SimpleMHsampler(θ) = SimpleMHsampler(θ, 0.0, 1)

mutable struct AdaptiveMHsampler <: MHsampler
    θ ::Vector{Float64}
    μ ::Vector{Float64}
    Σ
    log_posterior_val::Float64
    step::Int64
end
AdaptiveMHsampler(θ, μ, Σ) = AdaptiveMHsampler(θ, μ, Σ, 0.0, 1)

function step!(mhsampler::SimpleMHsampler, m::StatisticalModel)
    # Update the log posterior value of the model to its dataset (with specified prior) at current parameter θ
    mhsampler.log_posterior_val = log_posterior(m, mhsampler.θ)

    σ = 1/mhsampler.step    # Sigma value determines covariance, which decreases  
    # Calculate θ' from proposal distribution
    θ_prime = rand(MvNormal(mhsampler.θ, σ*I))

    # Initialize models with parameters θ_k and θ_prime 
    proposal_log_prob = log_posterior(m, θ_prime)

    logU = log(rand())      # Uniform[0, 1]
    
    if logU < min(0, proposal_log_prob - mhsampler.log_posterior_val) 
        mhsampler.θ = θ_prime           # Update to proposed state
    else logU ≥ min(0, proposal_log_prob - mhsampler.log_posterior_val)
        # mhsampler.θ = mhsampler.θ       # Keep state the same
    end 

    # Update step 
    mhsampler.step += 1

end

function step!(mhsampler::AdaptiveMHsampler, m::StatisticalModel)
    # Update the log posterior value of the model to its dataset (with specified prior) at current parameter θ
    mhsampler.log_posterior_val = log_posterior(m, mhsampler.θ)

    # Calculate θ' from proposal distribution
    θ_prime = rand(MvNormal(mhsampler.θ, mhsampler.Σ))

    # Initialize models with parameters θ_k and θ_prime 
    proposal_log_prob = log_posterior(m, θ_prime)

    logU = log(rand())      # Uniform[0, 1]
    
    if logU < min(0, proposal_log_prob - mhsampler.log_posterior_val) 
        mhsampler.θ = θ_prime           # Update to proposed state
    else logU ≥ min(0, proposal_log_prob - mhsampler.log_posterior_val)
        # mhsampler.θ = mhsampler.θ       # Keep state the same
    end 

    # Update step 
    mhsampler.step += 1

    # Update mean and covariance 
    # Mean: μ_{k+1} = μ_k + \frac{1}{k+1} (θ_{k+1} - μ_k) 
    # Covariance: Σ_{k+1} = Σ_k + \frac{1}{k+1} [(θ_{k+1} - μ_k)(θ_{k+1} - μ_k)^T - Σ_k]
    mhsampler.Σ = mhsampler.Σ + (1/(mhsampler.step +1))*(((mhsampler.θ - mhsampler.μ) * transpose(mhsampler.θ - mhsampler.μ)) - mhsampler.Σ)
    mhsampler.μ = mhsampler.μ + (1/(mhsampler.step +1))*(mhsampler.θ - mhsampler.μ)     # Must be updated second 

end

function run!(sampler::MHsampler, stat_model, nsteps::Int64, outp::MHoutp)
    push!(outp.θ_steps, sampler.θ)          # Should implement feed function? 
    for _ in 1:nsteps 
        step!(sampler, stat_model);
        push!(outp.θ_steps, sampler.θ)
        println(sampler.log_posterior_val)
    end 
end 


end