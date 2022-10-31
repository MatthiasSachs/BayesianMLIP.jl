module globalSamplers
using Distributions, LinearAlgebra
using BayesianMLIP.NLModels, BayesianMLIP.Utils, BayesianMLIP.Outputschedulers 
export State_θ, sampler 
export BAOAB_θ, BADODAB_θ, SGLD_θ
export MCMCsampler, MetropolisHastings, GibbsSampler
export step!, run! 


mutable struct State_θ
    θ
    θ_prime 
end 
State_θ(θ) = State_θ(θ, zeros(length(θ)))


abstract type sampler end 
abstract type globalSampler <: sampler end 
abstract type BAOsampler <: globalSampler end 
abstract type MCMCsampler <: globalSampler end 


# Non-Gradient MCMC Samplers 

mutable struct MetropolisHastings <: MCMCsampler 
    # Metropolis Hastings over both linear and nonlinear spaces
    h::Float64      # step size
    log_post_val::Float64 
    μ ::Vector{Float64}
    correctionΣ     # correction component of covariance matrix (2K × 2K)
    preconΣ         # preconditioned covariance matrix (2K × 2K)
    std             # std dev matrix 
    t::Int64        # step index 
    n_accepted::Int64 
    lp              # log posterior calculation function
    α::Float64      # damping term, α=1.0 makes this a nonadaptive sampler. 
    function MetropolisHastings(h::Float64, st::State_θ, stm::StatisticalModel, μ, precision, α=0.9)

        # Check that μ is K-vector and precision is K × K matrix
        twoK = nparams(stm)
        @assert(length(μ) == twoK) 
        @assert(size(precision) == (twoK, twoK))  

        std = precision_to_stddev(precision)
        cov = std * transpose(std) 

        # get log posterior value of initial state 
        lp = get_lp(stm, μ, std) 
        initial_lp = lp(st.θ, stm.data)

        new(h, initial_lp, μ, zeros(twoK, twoK), cov, std, 1, 0, lp, α);
    end 
end 

function step!(st::State_θ, s::MetropolisHastings, stm::StatisticalModel) 
    
    θ_proposal = st.θ + sqrt(s.h) * randn(length(st.θ))
    proposal_log_prob = s.lp(θ_proposal, stm.data) 
    
    print(string(s.log_post_val) * " --?--> " * string(proposal_log_prob) * " : ")

    if log(rand()) < min(0, proposal_log_prob - s.log_post_val) # Accept 
        st.θ = θ_proposal                                       # Update to proposed state
        s.log_post_val = proposal_log_prob                      # Update log_posterior value at new θ
        s.n_accepted += 1 
    end 

    println(s.log_post_val)

    # update correction covariance matrix at every step 
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(((st.θ - s.μ) * transpose(st.θ - s.μ)) - s.correctionΣ)
    
    # update mean 
    s.μ = s.μ + (1/(s.t))*(st.θ - s.μ)
    s.t += 1 

    # update std dev matrix after 100 steps and every 10 steps. 
    if s.t > 100 && s.t % 10 == 0 
        s.std = covariance_to_stddev(s.α * s.preconΣ + (1 - s.α) * s.correctionΣ)
        # s.lp = get_lp(stm, s.μ, s.std) 
    end 

    return s.log_post_val
end 

function run!(st::State_θ, s::MetropolisHastings, stm::StatisticalModel, Nsteps::Int64, outp; trueΣ = nothing)
    progress = length(outp.θ)

    if progress == 0        # for when we only reset the outp but maintain the adapted parameters
        s.n_accepted = 0
    end 

    for i in 1 + progress:Nsteps + progress
        print(i, "/", Nsteps + progress, ") ")
        step!(st, s, stm)

        # Push θ, log posterior value, rejection rate
        push!(outp.θ, st.θ)
        push!(outp.log_posterior, s.log_post_val)
        push!(outp.acceptance_rate, s.n_accepted/s.t)

        # Push condition number 
        cov = s.α * s.preconΣ + (1 - s.α) * s.correctionΣ
        eigenvalues = eigen(cov).values 
        minmax_ratio = maximum(eigenvalues)/minimum(eigenvalues)
        push!(outp.eigen_ratio, minmax_ratio)

        push!(outp.covariance_metric, norm(cov))
           
    end 
end 


mutable struct SGLD_θ <: sampler 
    h::Float64 
    F::Vector{Float64}
    β::Float64 
    mb_size::Int64
    glp
    lp
end 
SGLD_θ(h::Float64, state::State_θ, stm::StatisticalModel, mb_size::Int64, β::Float64=1.0) = SGLD_θ(h, get_glp(stm)(state.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data)), β, mb_size, get_glp(stm), get_lp(stm))

function step!(st::State_θ, s::SGLD_θ, stm::StatisticalModel)
    st.θ = st.θ + s.h * s.F + sqrt(2 * s.h / s.β) * randn(nparams(stm))
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
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


mutable struct BAOAB_θ <: BAOsampler
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


mutable struct BADODAB_θ <: BAOsampler
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



end