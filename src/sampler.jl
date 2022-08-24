module Samplers
using Distributions
using BayesianMLIP.NLModels
using BayesianMLIP.Utils
# using BayesianMLIP.MiniACEflux: FluxPotential
using LinearAlgebra
using LinearAlgebra: norm
using BayesianMLIP.Outputschedulers

export BAOSplitting, MHsampler, SimpleMHsamplers, sampler 
export State_θ, BAOAB_θ, BADODAB_θ, SGLD_θ, run! 
export SimpleMHsampler, AdaptiveMHsampler, step!, run!

mutable struct State_θ
    θ
    θ_prime 
end 

abstract type sampler end 
abstract type BAOSplitting <: sampler end 
abstract type MHsampler <: sampler end 


mutable struct SGLD_θ <: sampler 
    h::Float64 
    F::Vector{Float64}
    β::Float64 
    mb_size::Int64
    glp
    lp
end 
SGLD_θ(h::Float64, state::State_θ, stm::StatisticalModel, mb_size::Int64 ; β::Float64=1.0) = SGLD_θ(h, get_glp(stm)(state.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data)), β, mb_size, get_glp(stm), get_lp(stm))

function step!(st::State_θ, s::SGLD_θ, stm::StatisticalModel)
    st.θ = st.θ + s.h * s.F + sqrt(2 * s.h / s.β) * randn(length(st.θ))
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    s.F[1] = 4e5 * s.F[1]
end 

function run!(st::State_θ, s::SGLD_θ, stm::StatisticalModel, Nsteps::Int64, outp)
    
    # print first log_posterior value out 
    x = s.lp(st.θ, stm.data, length(stm.data))
    push!(outp.log_posterior, x)
    println("0) $x")

    for i in 1:Nsteps 
        first = st.θ
        step!(st, s, stm) 

        push!(outp.θ, st.θ)

        x = s.lp(st.θ, stm.data, length(stm.data))
        push!(outp.log_posterior, x)

        step_diff = norm(st.θ - first)

        println("$i) $x  ($step_diff)")       
    end 
end 


mutable struct BAOAB_θ <: BAOSplitting
    h::Float64      # Step size
    F::Vector{Float64}
    β::Float64 
    γ::Float64 
    mb_size::Int64
    glp
end 
BAOAB_θ(h::T, state::State_θ, stm::StatisticalModel, mb_size::Int64 ; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOAB_θ(h, get_glp(stm)(state.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data)), β, γ, mb_size, get_glp(stm))

function step!(st::State_θ, s::BAOAB_θ, stm::StatisticalModel) 
    st.θ_prime += 0.5 * s.h * s.F
    st.θ += 0.5 * s.h * st.θ_prime 
    st.θ_prime = exp(-s.h * s.γ) * st.θ_prime + sqrt((1/s.β) * (1 - exp(-2*s.γ*s.h))) * randn(length(st.θ_prime)) 
    st.θ += 0.5 * s.h * st.θ_prime
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    st.θ_prime += 0.5 * s.h * s.F
end 


mutable struct BADODAB_θ <: BAOSplitting
    h::Float64 
    F::Vector{Float64}
    β::Float64 
    n::Int64
    σG::Float64
    σA::Float64
    μ::Float64
    ξ::Float64
    mb_size::Int64
    glp 
end 
BADODAB_θ(h::Float64, state::State_θ, stm::StatisticalModel, mb_size::Int64 ; β=1.0, n=length(state.θ), σG=1.0, σA=1.0, μ=10.0, ξ=1.0) = BADODAB_θ(h, get_glp(stm)(state.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data)), β, n, σG, σA, μ, ξ, mb_size, get_glp(stm))

function step!(st::State_θ, s::BADODAB_θ, stm::StatisticalModel) 
    st.θ_prime += 0.5 * s.h * (s.F + s.σG .* randn(length(st.θ)))
    st.θ += 0.5 * s.h * st.θ_prime 
    s.ξ += 0.5 * s.h * (1/s.μ) * (dot(st.θ_prime, st.θ_prime) - s.n * (1/s.β))

    if -0.001 < s.ξ < 0.001 
        st.θ_prime = st.θ_prime + sqrt(s.h) * s.σA .* randn(length(st.θ))
    else 
        α = 1 - exp(-2 * s.ξ * s.h) 
        ζ = 2 * s.ξ
        st.θ_prime = exp(-s.ξ * s.h) * st.θ_prime + s.σA * sqrt(α/ζ) * randn(length(st.θ))
    end 

    s.ξ += 0.5 * s.h * (1/s.μ) * (dot(st.θ_prime, st.θ_prime) - s.n * (1/s.β))
    st.θ += 0.5 * s.h * st.θ_prime 
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    st.θ_prime += 0.5 * s.h * (s.F + s.σG .* randn(length(st.θ)))
end 

function run!(st::State_θ, s::BAOAB_θ, stm::StatisticalModel, Nsteps::Int64, outp) 
    for i in 1:Nsteps 
        step!(st, s, stm) 
        push!(outp.θ, st.θ)
        push!(outp.θ_prime, st.θ_prime)
        x = s.glp(st.θ, stm.data, length(stm.data))
        push!(outp.log_posterior, x)

        println(x)        # print log_posterior value 

        if i % 10 == 0
            println(i, "/", Nsteps)
        end 
    end 
end 

function run!(st::State_θ, s::BADODAB_θ, stm::StatisticalModel, Nsteps::Int64, outp) 
    for i in 1:Nsteps 
        step!(st, s, stm) 
        push!(outp.θ, st.θ)
        push!(outp.θ_prime, st.θ_prime)
        x = log_posterior(stm)
        push!(outp.log_posterior, x)
        push!(outp.ξ, s.ξ)

        print(x)        # print log_posterior value 

        if i % 10 == 0
            println(i, "/", Nsteps)
        end 
    end 
end 

# Non-Gradient MCMC Samplers 

mutable struct SimpleMHsampler <: MHsampler 
    h::Float64 
    log_post_val::Float64 
    Σ           # covariance matrix (shouldn't be changing)
    n_rejected::Int64
    lp
end 
SimpleMHsampler(h::Float64, st::State_θ, stm::StatisticalModel) = SimpleMHsampler(h, get_lp(stm)(st.θ, stm.data, length(stm.data)), I, 0, get_lp(stm))

function step!(st::State_θ, s::SimpleMHsampler, stm::StatisticalModel) 

    # Propose a new step and calculate its log posterior (no minibatch)
    θ_proposal = rand(MvNormal(st.θ, s.h * s.Σ)) 
    proposal_log_prob = s.lp(θ_proposal, stm.data, length(stm.data)) 
    
    print(string(s.log_post_val) * " --?--> " * string(proposal_log_prob) * " : ")

    if log(rand()) < min(0, proposal_log_prob - s.log_post_val) # Accept 
        st.θ = θ_proposal                                       # Update to proposed state
        s.log_post_val = proposal_log_prob                      # Update log_posterior value at new θ
    else                                                        # Reject 
        s.n_rejected += 1 
    end 

    return s.log_post_val
end 

mutable struct AdaptiveMHsampler <: MHsampler 
    h::Float64 
    log_post_val::Float64 
    μ ::Vector{Float64}
    Σ       # covariance matrix 
    Γ       # shadow covariance matrix 
    t::Int64    # step index 
    n_rejected::Int64 
    lp
end 
AdaptiveMHsampler(h::Float64, st::State_θ, stm::StatisticalModel, μ, Σ) = AdaptiveMHsampler(h, get_lp(stm)(st.θ, stm.data, length(stm.data)), μ, Σ, Σ, 1, 0, get_lp(stm)) 

function step!(st::State_θ, s::AdaptiveMHsampler, stm::StatisticalModel) 

    # Propose a new step and calculate its log posterior (no minibatch)
    θ_proposal = rand(MvNormal(st.θ, s.h * s.Σ)) 
    proposal_log_prob = s.lp(θ_proposal, stm.data, length(stm.data)) 
    
    print(string(s.log_post_val) * " --?--> " * string(proposal_log_prob) * " : ")

    if log(rand()) < min(0, proposal_log_prob - s.log_post_val) # Accept 
        st.θ = θ_proposal                                       # Update to proposed state
        s.log_post_val = proposal_log_prob                      # Update log_posterior value at new θ
    else                                                        # Reject 
        s.n_rejected += 1 
    end 

    println(s.log_post_val)

    # update shadow covariance matrix at every step 
    s.Γ = s.Γ + (1/(s.t +1))*(((st.θ - s.μ) * transpose(st.θ - s.μ)) - s.Γ)
    s.μ = s.μ + (1/(s.t +1))*(st.θ - s.μ)
    s.t += 1 

    # update true covariance after 100 steps and every 10 steps. 
    if s.t > 100 && s.t % 10 == 0 
        s.Σ = s.Γ 
    end 
    return s.log_post_val
end 

function run!(st::State_θ, s::MHsampler, stm::StatisticalModel, Nsteps::Int64, outp; trueΣ = nothing)
    progress = length(outp.θ)
    for i in 1 + progress:Nsteps + progress
        print(i, "/", Nsteps + progress, ") ")
        step!(st, s, stm)

        # Push θ, log posterior value, rejection rate
        push!(outp.θ, st.θ)
        push!(outp.log_posterior, s.log_post_val)
        push!(outp.rejection_rate, s.n_rejected/i)

        # Push condition number 
        eigenvalues = eigen(s.Σ).values 
        minmax_ratio = maximum(eigenvalues)/minimum(eigenvalues)
        push!(outp.eigen_ratio, minmax_ratio)

        if trueΣ !== nothing 
            # Push covariance metric 
            push!(outp.covariance_metric, norm(s.Σ - trueΣ)) 
        end 
           
    end 
end 

end