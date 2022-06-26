module Samplers
using Distributions
using BayesianMLIP.MHoutputschedulers
using BayesianMLIP.NLModels
using BayesianMLIP.ACEflux: FluxPotential
export SimpleMHsampler, AdaptiveMHsampler, step!, run!


abstract type sampler end 
abstract type MHsampler end 

mutable struct SimpleMHsampler <: MHsampler     # Should fieldnames be this? 
    θ ::Vector{Float64}
    μ ::Vector{Float64}
    Σ
    step::Int64
    weight_E::Float64
    weight_F::Float64 
    log_likelihood
    prior 
    log_posterior 
end 

mutable struct AdaptiveMHsampler <: MHsampler   # Should fieldnames be this? 
    θ ::Vector{Float64}
    μ ::Vector{Float64}
    Σ
    step::Int64
end
AdaptiveMHsampler(θ, μ, Σ) = AdaptiveMHsampler(θ, μ, Σ, 1)

function step!(mhsampler::SimpleMHsampler, model, log_posterior)
    # Calculate θ' from proposal distribution
    θ_prime = vec(rand(MvNormal(mhsampler.θ, mhsampler.Σ), 1))

    # Initialize models with parameters θ_k and θ_prime 
    model1 = deepcopy(model);
    model2 = deepcopy(model);
    set_params!(model1, mhsampler.θ)    # Set parameters θ
    set_params!(model2, θ_prime)        # Set parameters θ'

    a = log_posterior(model1, data, prior, log_likelihood)  # Calculate log_posterior to get acceptance probability
    b = log_posterior(model2, data, prior, log_likelihood)  # Calculate log_posterior to get acceptance probability

    logU = log(rand())      # Uniform[0, 1]
    
    if logU < min(0, a - b) 
        mhsampler.θ = θ_prime           # Update to proposed state
    else logU ≥ min(0, a - b)
        # mhsampler.θ = mhsampler.θ       # Keep state the same
    end 

    # Update step 
    mhsampler.step += 1

end

function step!(mhsampler::AdaptiveMHsampler, model, log_posterior)
    # Calculate θ' from proposal distribution
    θ_prime = vec(rand(MvNormal(mhsampler.θ, mhsampler.Σ), 1))

    # Initialize models with parameters θ_k and θ_prime 
    model1 = deepcopy(model);
    model2 = deepcopy(model);
    set_params!(model1, mhsampler.θ) 
    set_params!(model2, θ_prime) 

    a = log_posterior(model1, data, prior, log_likelihood)
    b = log_posterior(model2, data, prior, log_likelihood)

    logU = log(rand())
    
    if logU < min(0, a - b) 
        mhsampler.θ = θ_prime           # Update to proposed state
    else logU ≥ min(0, a - b)
        # mhsampler.θ = mhsampler.θ       # Keep state the same
    end 

    # Update mean and covariance 
    # Mean: μ_{k+1} = μ_k + \frac{1}{k+1} (θ_{k+1} - μ_k) 
    # Covariance: Σ_{k+1} = Σ_k + \frac{1}{k+1} [(θ_{k+1} - μ_k)(θ_{k+1} - μ_k)^T - Σ_k]
    mhsampler.Σ = mhsampler.Σ + (1/(mhsampler.step +1))*(((mhsampler.θ - mhsampler.μ) * transpose(mhsampler.θ - mhsampler.μ)) - mhsampler.Σ)
    mhsampler.μ = mhsampler.μ + (1/(mhsampler.step +1))*(mhsampler.θ - mhsampler.μ)     # Must be updated second 
    
    # Update step 
    mhsampler.step += 1

end

function run!(sampler::MHsampler, nsteps::Int64, model, log_posterior, outp::MHoutp)
    push!(outp.θ_steps, sampler.θ)          # Should implement feed function? 
    for _ in 1:nsteps 
        step!(sampler, model, log_posterior);
        push!(outp.θ_steps, sampler.θ)
    end 
end 


end