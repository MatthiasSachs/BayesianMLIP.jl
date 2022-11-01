module conditionalSamplers 
using Distributions, LinearAlgebra
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Utils, BayesianMLIP.Outputschedulers, BayesianMLIP.globalSamplers
using BayesianMLIP.globalSamplers: sampler, State_θ, MCMCsampler
export step!, run! 
export linearMetropolis, nonlinearMetropolis, GibbsSampler
export linearSGLD, nonlinearSGLD, linearStochasticGD
export GibbsSampler

abstract type conditionalSampler <: sampler end 
abstract type cMCMCSampler <: conditionalSampler end 
abstract type cGradientSampler <: conditionalSampler end 


mutable struct GibbsSampler
    # Gibbs update algorithm for FS model 
    linear_sampler::conditionalSampler 
    nonlinear_sampler::conditionalSampler
    lin_update_prob::Float64
    t::Int64
    function GibbsSampler(linear_sampler::conditionalSampler, nonlinear_sampler::conditionalSampler, lin_update_prob::Float64) 
        new(linear_sampler, nonlinear_sampler, lin_update_prob, 1)
    end
end 

function step!(st::State_θ, s::GibbsSampler, stm::StatisticalModel)
    if rand() < s.lin_update_prob
        step!(st, s.linear_sampler, stm)
        println(st.θ)
        s.nonlinear_sampler.log_post_val = s.linear_sampler.log_post_val
    else 
        step!(st, s.nonlinear_sampler, stm)
        println(st.θ)
        s.linear_sampler.log_post_val = s.nonlinear_sampler.log_post_val
    end 

    s.t += 1 
end 

function run!(st::State_θ, s::GibbsSampler, stm::StatisticalModel, Nsteps::Int64, outp; trueΣ = nothing)
    progress = length(outp.θ)

    # if progress == 0        # for when we only reset the outp but maintain the adapted parameters
    #     s.n_accepted = 0
    # end 

    for i in 1 + progress:Nsteps + progress
        print(i, "/", Nsteps + progress, ") ")
        step!(st, s, stm)

        # Push θ, log posterior value, rejection rate
        push!(outp.θ, st.θ)
        # push!(outp.log_posterior, s.log_post_val)
        # push!(outp.acceptance_rate, s.n_accepted/s.t)

        # Push condition number 
        # cov = s.α * s.preconΣ + (1 - s.α) * s.correctionΣ
        # eigenvalues = eigen(cov).values 
        # minmax_ratio = maximum(eigenvalues)/minimum(eigenvalues)
        # push!(outp.eigen_ratio, minmax_ratio)

        # push!(outp.covariance_metric, norm(cov))
           
    end 
end 

# ------------------------------- Metropolis Hastings -------------------------------

mutable struct linearMetropolis <: cMCMCSampler
    h 
    lp 
    log_post_val
    μ
    correctionΣ
    preconΣ
    std
    t 
    n_accepted 
    α 
    transf_mean 
    transf_std
    function linearMetropolis(h::Float64, st::State_θ, stm::StatisticalModel; α=0.9) 
        K = nlinparams(stm)  

        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]

        lp = get_lp(stm, transf_mean, transf_std) 
        initial_lp = lp(st.θ, stm.data, length(stm.data))

        new(h, lp, initial_lp, zeros(K), zeros(K, K), Diagonal(ones(K)), Diagonal(ones(K)), 1, 0, α, transf_mean, transf_std);  
    end 

    function linearMetropolis(h::Float64, st::State_θ, stm::StatisticalModel, transf_mean, transf_std; α=0.9) 
        twoK = nparams(stm) 
        @assert length(transf_mean) == twoK 
        @assert size(transf_std) == (twoK, twoK)

        K = nlinparams(stm) 
        lp = get_lp(stm, transf_mean, transf_std) 
        initial_lp = lp(st.θ, stm.data, length(stm.data))

        new(h, lp, initial_lp, zeros(K), zeros(K, K), Diagonal(ones(K)), Diagonal(ones(K)), 1, 0, α)
    end 
end 

function step!(st::State_θ, s::linearMetropolis, stm::StatisticalModel) 
    K = nlinparams(stm)
    # only update the linear component of st.θ
    θ_proposal = st.θ + sqrt(s.h) *  vcat(s.std * randn(K), zeros(K))
    proposal_log_prob = s.lp(θ_proposal, stm.data, length(stm.data)) 

    print(string(s.log_post_val) * " --l?--> " * string(proposal_log_prob) * " : ")

    if log(rand()) < min(0, proposal_log_prob - s.log_post_val)
        st.θ = θ_proposal                                      
        s.log_post_val = proposal_log_prob                     
        s.n_accepted += 1 
    end 

    println(s.log_post_val)

    # update linear correction covariance matrix at every step 
    lin_comp_of_state = st.θ[1:K]
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (lin_comp_of_state - s.μ) * transpose(lin_comp_of_state - s.μ)) - s.correctionΣ)
    
    # update linear mean 
    s.μ = s.μ + (1/(s.t)) * (lin_comp_of_state - s.μ)

    # update linear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 100 && s.t % 10 == 0 
        s.std = covariance_to_stddev(s.α * s.preconΣ + (1 - s.α) * s.correctionΣ)
    end 

    s.t += 1 

    return s.log_post_val

end 

mutable struct nonlinearMetropolis <: cMCMCSampler
    h 
    lp 
    log_post_val
    μ
    correctionΣ
    preconΣ
    std
    t 
    n_accepted 
    α 
    function nonlinearMetropolis(h::Float64, st::State_θ, stm::StatisticalModel; α=0.9) 

        K = nlinparams(stm)  

        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]

        lp = get_lp(stm, transf_mean, transf_std) 
        initial_lp = lp(st.θ, stm.data, length(stm.data))

        new(h, lp, initial_lp, zeros(K), zeros(K, K), Diagonal(ones(K)), Diagonal(ones(K)), 1, 0, α);  
    end 
end 

function step!(st::State_θ, s::nonlinearMetropolis, stm::StatisticalModel) 
    K = nlinparams(stm)
    # only update the nonlinear component of st.θ
    θ_proposal = st.θ + sqrt(s.h) *  vcat(zeros(K), s.std * randn(K))
    proposal_log_prob = s.lp(θ_proposal, stm.data, length(stm.data)) 

    print(string(s.log_post_val) * " --n?--> " * string(proposal_log_prob) * " : ")

    if log(rand()) < min(0, proposal_log_prob - s.log_post_val)
        st.θ = θ_proposal                                      
        s.log_post_val = proposal_log_prob                     
        s.n_accepted += 1 
    end 

    println(s.log_post_val)

    

    # update linear correction covariance matrix at every step 
    nlin_comp_of_state = st.θ[K+1:end]
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (nlin_comp_of_state - s.μ) * transpose(nlin_comp_of_state - s.μ)) - s.correctionΣ)
    
    # update linear mean 
    s.μ = s.μ + (1/(s.t)) * (nlin_comp_of_state - s.μ)

    # update linear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 100 && s.t % 10 == 0 
        s.std = covariance_to_stddev(s.α * s.preconΣ + (1 - s.α) * s.correctionΣ)
    end 

    s.t += 1 

    return s.log_post_val

end 

function run!(st::State_θ, s::cMCMCSampler, stm::StatisticalModel, Nsteps::Int64, outp; trueΣ = nothing)
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

# ------------------------------- Stochastic GD -------------------------------

mutable struct linearStochasticGD <: cGradientSampler 
    h 
    lp 
    glp
    log_post_val 
    F 
    β 
    mb_size 
    t 
    α
    function linearStochasticGD(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64; β::Float64=1.0, α::Float64=0.9)
        @assert mb_size <= length(stm.data) 
        K = nlinparams(stm)

        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]

        lp = get_lp(stm, transf_mean, transf_std) 
        glp = get_glp(stm, transf_mean, transf_std)

        initial_lp = lp(st.θ, stm.data, length(stm.data))

        F = glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data))
        # F[1:K] = F[1:K] .* [1e7, 1e4, 1]
        F[K+1:end] = zeros(K) 

        new(h, lp, glp, initial_lp, F, β, mb_size, 1, α)
    end 
end 

function step!(st::State_θ, s::linearStochasticGD, stm::StatisticalModel)
    K = nlinparams(stm)

    st.θ -= s.h * s.F
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    # s.F[1:K] = s.F[1:K] .* [1e7, 1e4, 1]
    s.F[K+1:end] = zeros(K) 
    println(s.F)
    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data))

    # println(string(s.log_post_val) * " --*--> " * string(new_log_post_val) * " : " * string(new_log_post_val))

    s.log_post_val = new_log_post_val

    s.t += 1 
end 

function run!(st::State_θ, s::linearStochasticGD, stm::StatisticalModel, Nsteps::Int64, outp) 
    progress = length(outp.θ)

    if progress == 0        # for when we only reset the outp but maintain the adapted parameters
        # s.n_accepted = 0
    end 

    for i in 1 + progress:Nsteps + progress
        print(i, "/", Nsteps + progress, ") ")
        step!(st, s, stm)

        # Push θ, log posterior value, rejection rate
        push!(outp.θ, st.θ)
        push!(outp.log_posterior, s.log_post_val)
        push!(outp.F1, s.F[1])
        push!(outp.F2, s.F[2])
        push!(outp.F3, s.F[3])
           
    end 
end 

# ----------------------------------- SGLD -----------------------------------

mutable struct linearSGLD <: cGradientSampler 
    h 
    lp 
    glp
    log_post_val 
    F 
    β 
    mb_size 
    μ
    correctionΣ 
    preconΣ 
    std 
    t 
    α
    function linearSGLD(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64, preconΣ=Diagonal(ones(nlinparams(stm))); β::Float64=1.0, α::Float64=0.9)
        @assert mb_size <= length(stm.data) 
        K = nlinparams(stm)

        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]

        lp = get_lp(stm, transf_mean, transf_std) 
        glp = get_glp(stm, transf_mean, transf_std)

        initial_lp = lp(st.θ, stm.data, length(stm.data))

        F = glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data))
        F[1:K] = F[1:K] .* [3.16e6, 1.78e3, 1]
        F[K+1:end] = zeros(K) 

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        std = Diagonal(ones(K)) 

        new(h, lp, glp, initial_lp, F, β, mb_size, μ, correctionΣ, preconΣ, std, 1, α)
    end 
end 

function step!(st::State_θ, s::linearSGLD, stm::StatisticalModel)
    # step size 1e-11
    K = nlinparams(stm)

    st.θ -= s.h * s.F + sqrt(2 * s.h / s.β) * vcat(s.std * randn(K), zeros(K))
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    s.F[1:K] = s.F[1:K] .* [3.16e6, 1.78e3, 1]
    s.F[K+1:end] = zeros(K) 
    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data))

    println(string(s.log_post_val) * " --*--> " * string(new_log_post_val) * " : " * string(new_log_post_val))

    s.log_post_val = new_log_post_val

    # update linear correction covariance matrix at every step 
    lin_comp_of_state = st.θ[1:K]
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (lin_comp_of_state - s.μ) * transpose(lin_comp_of_state - s.μ)) - s.correctionΣ)
    
    # update linear mean 
    s.μ = s.μ + (1/(s.t)) * (lin_comp_of_state - s.μ)

    # update linear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 100 && s.t % 10 == 0 
        s.std = covariance_to_stddev(s.α * s.preconΣ .+ (1 - s.α) * s.correctionΣ)
    end 

    s.t += 1 
end 

mutable struct nonlinearSGLD <: cGradientSampler 
    h 
    lp 
    glp
    log_post_val 
    F 
    β 
    mb_size 
    μ
    correctionΣ 
    preconΣ 
    std 
    t 
    α
    function nonlinearSGLD(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64, preconΣ=Diagonal(ones(nlinparams(stm))); β::Float64=1.0, α::Float64=0.9)
        @assert mb_size <= length(stm.data) 
        K = nlinparams(stm)

        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]

        lp = get_lp(stm, transf_mean, transf_std) 
        glp = get_glp(stm, transf_mean, transf_std)

        initial_lp = lp(st.θ, stm.data, length(stm.data))

        F = glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data))
        F[1:K] = zeros(K)
        F[K+1:end] = F[K+1:end] .* [1e13, 3.16e6, 1]

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        std = Diagonal(ones(K)) 

        new(h, lp, glp, initial_lp, F, β, mb_size, μ, correctionΣ, preconΣ, std, 1, α)
    end 
end 

function step!(st::State_θ, s::nonlinearSGLD, stm::StatisticalModel)
    K = nlinparams(stm)

    st.θ -= s.h * s.F + sqrt(2 * s.h / s.β) * vcat(zeros(K), s.std * randn(K))
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    s.F[1:K] = zeros(K)
    s.F[K+1:end] = s.F[K+1:end] .* [1e13, 3.16e6, 1]
    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data))

    println(string(s.log_post_val) * " --*--> " * string(new_log_post_val) * " : " * string(new_log_post_val))
    println(s.F)

    s.log_post_val = new_log_post_val

    # update nonlinear correction covariance matrix at every step 
    nlin_comp_of_state = st.θ[K+1:end]
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (nlin_comp_of_state - s.μ) * transpose(nlin_comp_of_state - s.μ)) - s.correctionΣ)
    
    # update nonlinear mean 
    s.μ = s.μ + (1/(s.t)) * (nlin_comp_of_state - s.μ)

    # update nonlinear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 100 && s.t % 10 == 0 
        s.std = covariance_to_stddev(s.α * s.preconΣ .+ (1 - s.α) * s.correctionΣ)
    end 

    s.t += 1 
end 

function run!(st::State_θ, s::Union{linearSGLD, nonlinearSGLD}, stm::StatisticalModel, Nsteps::Int64, outp) 
    progress = length(outp.θ)

    if progress == 0        # for when we only reset the outp but maintain the adapted parameters
        # s.n_accepted = 0
    end 

    for i in 1 + progress:Nsteps + progress
        print(i, "/", Nsteps + progress, ") ")
        step!(st, s, stm)

        # Push θ, log posterior value, rejection rate
        push!(outp.θ, st.θ)
        push!(outp.log_posterior, s.log_post_val)

        cov = s.α * s.preconΣ .+ (1 - s.α) * s.correctionΣ
        eigenvalues = eigen(cov).values 
        minmax_ratio = maximum(eigenvalues)/minimum(eigenvalues)
        push!(outp.eigen_ratio, minmax_ratio)

        push!(outp.covariance_metric, norm(cov))
           
    end 
end 

# ----------------------------------- BAOAB -----------------------------------

mutable struct linearBAOAB 

end 

end # end module 