module conditionalSamplers 
using Distributions, LinearAlgebra
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Utils, BayesianMLIP.Outputschedulers, BayesianMLIP.globalSamplers
using BayesianMLIP.globalSamplers: sampler, State_θ, MCMCsampler
export step!, run! 
export linearMetropolis, nonlinearMetropolis, GibbsSampler
export linearSGLD, nonlinearSGLD, linearStochasticGD, nonlinearSGLD2
export linearBAOAB, nonlinearBAOAB
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
    function linearMetropolis(h::Float64, st::State_θ, stm::StatisticalModel, α=0.9) 
        K = nlinparams(stm)  
    
        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]
    
        lp = get_lp(stm, transf_mean, transf_std) 
        initial_lp = lp(st.θ, stm.data, length(stm.data))
    
        new(h, lp, initial_lp, zeros(K), zeros(K, K), Diagonal(ones(K)), Diagonal(ones(K)), 1, 0, α, transf_mean, transf_std);  
    end 
    
    function linearMetropolis(h::Float64, st::State_θ, stm::StatisticalModel, α, transf_mean, transf_std)
        # for when you want to do a MANUAL change of basis 
        twoK = nparams(stm) 
        @assert length(transf_mean) == twoK 
        @assert size(transf_std) == (twoK, twoK)
    
        K = nlinparams(stm) 
        lp = get_lp(stm, transf_mean, transf_std) 
        initial_lp = lp(st.θ, stm.data, length(stm.data))
    
        new(h, lp, initial_lp, zeros(K), zeros(K, K), Diagonal(ones(K)), Diagonal(ones(K)), 1, 0, α, transf_mean, transf_std)
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
    transf_mean 
    transf_std 
    function nonlinearMetropolis(h::Float64, st::State_θ, stm::StatisticalModel, α) 

        K = nlinparams(stm)  

        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]

        lp = get_lp(stm, transf_mean, transf_std) 
        initial_lp = lp(st.θ, stm.data, length(stm.data))

        new(h, lp, initial_lp, zeros(K), zeros(K, K), Diagonal(ones(K)), Diagonal(ones(K)), 1, 0, α, transf_mean, transf_std);  
    end 

    function nonlinearMetropolis(h::Float64, st::State_θ, stm::StatisticalModel, α, transf_mean, transf_std) 
        twoK = nparams(stm) 
        @assert length(transf_mean) == twoK 
        @assert size(transf_std) == (twoK, twoK)

        K = nlinparams(stm)  

        lp = get_lp(stm, transf_mean, transf_std) 
        initial_lp = lp(st.θ, stm.data, length(stm.data))

        new(h, lp, initial_lp, zeros(K), zeros(K, K), Diagonal(ones(K)), Diagonal(ones(K)), 1, 0, α, transf_mean, transf_std);  
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

# -------------------------- Stochastic GD (SGLD with β=Inf) --------------------------

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
    transf_mean 
    transf_std 
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
        F[1:K] = F[1:K] .* [1e7, 1e4, 1]
        F[K+1:end] = zeros(K) 

        new(h, lp, glp, initial_lp, F, β, mb_size, 1, α, transf_mean, transf_std)
    end 

    function linearStochasticGD(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64, β::Float64, α::Float64, transf_mean, transf_std)
        @assert mb_size <= length(stm.data) 
        K = nlinparams(stm)
        @assert length(transf_mean) == 2K 
        @assert size(transf_std) == (2K, 2K)

        lp = get_lp(stm, transf_mean, transf_std) 
        glp = get_glp(stm, transf_mean, transf_std)

        initial_lp = lp(st.θ, stm.data, length(stm.data))

        F = glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data))
        # F[1:K] = F[1:K] .* [1e7, 1e4, 1]
        F[K+1:end] = zeros(K) 

        new(h, lp, glp, initial_lp, F, β, mb_size, 1, α, transf_mean, transf_std)
    end 
end 

function step!(st::State_θ, s::linearStochasticGD, stm::StatisticalModel)
    K = nlinparams(stm)

    st.θ -= s.h * s.F
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    s.F[1:K] = s.F[1:K] .* [1.78e6, 3.14e3, 1]
    s.F[K+1:end] = zeros(K) 
    # println(s.F)
    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data))

    println(string(s.log_post_val) * " --*--> " * string(new_log_post_val) * " : " * string(new_log_post_val))

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
    Σ
    std 
    t 
    α
    transf_mean 
    transf_std 
    function linearSGLD(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64; β::Float64=1.0, α::Float64=0.9)
        # linearSGLD(4e-11, st, stm, 1; β=1e-5, α=0.9) ;
        @assert mb_size <= length(stm.data) 
        K = nlinparams(stm)

        hp = precon_pre_cov_mean(stm) 
        transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
        transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]

        lp = get_lp(stm, transf_mean, transf_std) 
        glp = get_glp(stm, transf_mean, transf_std)

        initial_lp = lp(st.θ, stm.data, length(stm.data))

        F = glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data))
        F[K+1:end] = zeros(K); 

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        preconΣ=Diagonal(ones(nlinparams(stm)))
        Σ = α * preconΣ + (1 - α) * correctionΣ
        std = Diagonal(ones(K)) 

        new(h, lp, glp, initial_lp, F, β, mb_size, μ, correctionΣ, preconΣ, Σ,  std, 1, α, transf_mean, transf_std)
    end 
end 

function step!(st::State_θ, s::linearSGLD, stm::StatisticalModel)
    # step size 4e-11
    K = nlinparams(stm)

    st.θ -= s.h * [s.Σ zeros(K, K); zeros(K, K) zeros(K, K)] * s.F + sqrt(2 * s.h / s.β) * vcat(s.std * randn(K), zeros(K))
    
    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data)); s.F[K+1:end] = zeros(K) 

    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data))

    println(string(s.log_post_val) * " --*--> " * string(new_log_post_val) * " : " * string(new_log_post_val))
    println(s.h * [s.Σ zeros(K, K); zeros(K, K) zeros(K, K)] * s.F)
    println(sqrt(2 * s.h / s.β) * vcat(s.std * randn(K), zeros(K)))
    s.log_post_val = new_log_post_val

    # update linear correction covariance matrix at every step 
    lin_comp_of_state = st.θ[1:K]
    
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (lin_comp_of_state - s.μ) * transpose(lin_comp_of_state - s.μ)) - s.correctionΣ)

    # update linear mean 
    s.μ = s.μ + (1/(s.t)) * (lin_comp_of_state - s.μ)

    # update linear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 50 && s.t % 10 == 0 
        decreasing_α = s.α * (0.99999^s.t)
        s.Σ = decreasing_α * s.preconΣ .+ ((1 - decreasing_α) * s.correctionΣ)
        s.std = covariance_to_stddev(s.Σ)
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
    Σ
    Σsquared    # covariance squared, which is needed for adapative conditioning for NONLINEAR 
    std 
    t 
    α
    transf_mean 
    transf_std 
    function nonlinearSGLD(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64, preconΣ=Diagonal(ones(nlinparams(stm))); β::Float64=1.0, α::Float64=0.9)
        # nonlinearSGLD(5e-9, st, stm, 1; β=1e-5, α=0.9) ;
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

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        preconΣ=Diagonal(ones(nlinparams(stm)))
        Σ = α * preconΣ + (1 - α) * correctionΣ
        std = Diagonal(ones(K)) 
        Σsquared = Diagonal(ones(K))

        new(h, lp, glp, initial_lp, F, β, mb_size, μ, correctionΣ, preconΣ, Σ, Σsquared, std, 1, α, transf_mean, transf_std)
    end 
end 

function step!(st::State_θ, s::nonlinearSGLD, stm::StatisticalModel)
    # step size 
    K = nlinparams(stm)

    flow = s.h * [zeros(K, K) zeros(K, K); zeros(K, K) s.Σsquared] * s.F
    noise = sqrt(2 * s.h / s.β) * vcat(zeros(K), s.std * randn(K))

    st.θ -= flow + noise

    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data)); s.F[1:K] = zeros(K) 

    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data))

    println(string(s.log_post_val) * " --*--> " * string(new_log_post_val) * " : " * string(new_log_post_val))
    println(flow)
    println(noise)

    s.log_post_val = new_log_post_val

    # update nonlinear correction covariance matrix at every step 
    nlin_comp_of_state = st.θ[K+1:end]

    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (nlin_comp_of_state - s.μ) * transpose(nlin_comp_of_state - s.μ)) - s.correctionΣ)
    
    # update nonlinear mean 
    s.μ = s.μ + (1/(s.t)) * (nlin_comp_of_state - s.μ)

    # update nonlinear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 20 && s.t % 10 == 0 
        decreasing_α = s.α * (0.999999^s.t)
        s.Σ = decreasing_α * s.preconΣ .+ ((1 - decreasing_α) * s.correctionΣ)
        Σ_svd = svd(s.Σ)
        D = Σ_svd.U * Diagonal(Σ_svd.S) 
        s.Σsquared = D * transpose(D) 
        s.std = covariance_to_stddev(s.Σ)
    end 

    s.t += 1 
end 

mutable struct nonlinearSGLD2 <: cGradientSampler 
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
    Σ
    Σsquared    # covariance squared, which is needed for adapative conditioning for NONLINEAR 
    std 
    t 
    α
    transf_mean 
    transf_std 
    function nonlinearSGLD2(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64, preconΣ=Diagonal(ones(nlinparams(stm))); β::Float64=1.0, α::Float64=0.9)
        # nonlinearSGLD(5e-9, st, stm, 1; β=1e-5, α=0.9) ;
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

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        preconΣ=Diagonal(ones(nlinparams(stm)))
        Σ = α * preconΣ + (1 - α) * correctionΣ
        std = Diagonal(ones(K)) 
        Σsquared = Diagonal(ones(K))

        new(h, lp, glp, initial_lp, F, β, mb_size, μ, correctionΣ, preconΣ, Σ, Σsquared, std, 1, α, transf_mean, transf_std)
    end 
end 

function step!(st::State_θ, s::nonlinearSGLD2, stm::StatisticalModel)
    # step size 
    K = nlinparams(stm)

    flow = s.h * [zeros(K, K) zeros(K, K); zeros(K, K) s.Σ .* [3.16e6, 1.7e3, 1]] * s.F
    noise = sqrt(2 * s.h / s.β) * vcat(zeros(K), s.std * randn(K))

    st.θ -= flow + noise

    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data)); s.F[1:K] = zeros(K) 

    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data))

    println(string(s.log_post_val) * " --*--> " * string(new_log_post_val) * " : " * string(new_log_post_val))
    println(flow)
    println(noise)

    s.log_post_val = new_log_post_val

    # update nonlinear correction covariance matrix at every step 
    nlin_comp_of_state = st.θ[K+1:end]
    prev_μ = s.μ

    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (s.F[K+1:end] - s.μ) * transpose(s.F[K+1:end] - prev_μ)) - s.correctionΣ)
    
    # update nonlinear mean 
    s.μ = s.μ + (1/(s.t)) * (s.F[K+1:end] - s.μ)

    # update nonlinear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 20 && s.t % 10 == 0 
        decreasing_α = s.α * (0.99999^s.t)
        s.Σ = decreasing_α * s.preconΣ .+ ((1 - decreasing_α) * s.correctionΣ)
        Σ_svd = svd(s.Σ)
        D = Σ_svd.U * Diagonal(Σ_svd.S) 
        s.Σsquared = D * transpose(D) 
        s.Σsquared = Diagonal(ones(K))
        s.std = covariance_to_stddev(s.Σ)
    end 

    s.t += 1 
end 

function run!(st::State_θ, s::Union{linearSGLD, nonlinearSGLD, nonlinearSGLD2}, stm::StatisticalModel, Nsteps::Int64, outp) 
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

        eigenvalues = eigen(s.Σ).values 
        push!(outp.eigen_ratio, maximum(eigenvalues)/minimum(eigenvalues))

        push!(outp.covariance_metric, norm(s.Σ))
           
    end 
end 

# ----------------------------------- BAOAB -----------------------------------

mutable struct linearBAOAB <: cGradientSampler 
    h 
    lp 
    glp
    log_post_val 
    F 
    β 
    γ
    mb_size 
    μ
    correctionΣ 
    preconΣ 
    Σ       # covariance i.e. inverse mass
    std 
    t 
    α
    transf_mean 
    transf_std
    function linearBAOAB(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64; β::Float64=1.0, γ::Float64=1.0, α::Float64=0.9, transf_mean = nothing, transf_std = nothing)
        
        @assert mb_size <= length(stm.data) 
        K = nlinparams(stm)

        if transf_mean === nothing && transf_std === nothing
            hp = precon_pre_cov_mean(stm) 
            transf_mean = vcat(hp["lin_mean"], hp["nlin_mean"])
            transf_std = [precision_to_stddev(hp["lin_precision"]) zeros(K, K); zeros(K, K) precision_to_stddev(hp["nlin_precision"])]
        else 
            @assert length(transf_mean) == 2K 
            @assert size(transf_std) == (2K, 2K)
        end 

        lp = get_lp(stm, transf_mean, transf_std) 
        glp = get_glp(stm, transf_mean, transf_std)

        initial_lp = lp(st.θ, stm.data, length(stm.data))

        F = glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data))
        F[1:K] = F[1:K]
        F[K+1:end] = zeros(K) 

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        preconΣ = Diagonal(ones(nlinparams(stm))) 
        std = Diagonal(ones(K)) 
        Σ = Diagonal(ones(K))

        new(h, lp, glp, initial_lp, F, β, γ, mb_size, μ, correctionΣ, preconΣ, Σ, std, 1, α, transf_mean, transf_std)
    end 
end 

function step!(st::State_θ, s::linearBAOAB, stm::StatisticalModel) 
    K = nlinparams(stm)
    M = inv(s.Σ)    # mass is covariance
    Dec = svd(M)
    sqrtM = Symmetric(Dec.U * Diagonal(sqrt.(Dec.S)) * Dec.Vt)

    st.θ_prime -= 0.5 * s.h * s.F

    st.θ += 0.5s.h * [s.Σ zeros(K, K); zeros(K, K) zeros(K, K)] * st.θ_prime 

    st.θ_prime = exp(-s.h * s.γ) * st.θ_prime + sqrt((1/s.β) * (1 - exp(-2*s.γ*s.h))) * vcat(sqrtM * randn(K), zeros(K))

    st.θ += 0.5s.h * [s.Σ zeros(K, K); zeros(K, K) zeros(K, K)] * st.θ_prime 

    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    # s.F[1:K] = s.F[1:K] .* [1e7, 3e3, 1]
    s.F[K+1:end] = zeros(K)
    println(s.F)

    st.θ_prime -= 0.5 * s.h * s.F 

    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data)) 

    println(string(s.log_post_val) * " --l--> " * string(new_log_post_val) * " : " * string(new_log_post_val))

    s.log_post_val = new_log_post_val

    # update linear correction covariance matrix at every step 
    lin_comp_of_state = st.θ[1:K]
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (lin_comp_of_state - s.μ) * transpose(lin_comp_of_state - s.μ)) - s.correctionΣ)
    
    # update linear mean 
    s.μ = s.μ + (1/(s.t)) * (lin_comp_of_state - s.μ)

    # update linear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 100 && s.t % 10 == 0 
        s.Σ = s.α * s.preconΣ .+ (1 - s.α) * s.correctionΣ
        s.std = covariance_to_stddev(s.Σ)
    end 

    s.t += 1 
end 

mutable struct nonlinearBAOAB <: cGradientSampler 
    h 
    lp 
    glp
    log_post_val 
    F 
    β 
    γ
    mb_size 
    μ
    correctionΣ 
    preconΣ 
    std 
    Minv # should be diagonal entries of covariance
    t 
    α
    transf_mean 
    transf_std
    function nonlinearBAOAB(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64, β::Float64=1.0, γ::Float64=1.0, α::Float64=0.9)
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
        F[K+1:end] = F[K+1:end]

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        preconΣ=Diagonal(ones(nlinparams(stm))) 
        std = Diagonal(ones(K)) 
        Minv = Diagonal(ones(K))

        new(h, lp, glp, initial_lp, F, β, γ, mb_size, μ, correctionΣ, preconΣ, std, Minv, 1, α, transf_mean, transf_std)
    end 

    function nonlinearBAOAB(h::Float64, st::State_θ, stm::StatisticalModel, mb_size::Int64, β::Float64, γ::Float64, α::Float64, transf_mean, transf_std)
        @assert mb_size <= length(stm.data) 
        K = nlinparams(stm)

        lp = get_lp(stm, transf_mean, transf_std) 
        glp = get_glp(stm, transf_mean, transf_std)

        initial_lp = lp(st.θ, stm.data, length(stm.data))

        F = glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), mb_size, replace = false)], length(stm.data))
        F[1:K] = zeros(K) 
        F[K+1:end] = F[K+1:end]

        μ = zeros(K) 
        correctionΣ = zeros(K, K)
        preconΣ=Diagonal(ones(nlinparams(stm))) 
        std = Diagonal(ones(K)) 
        Minv = Diagonal(ones(K))

        new(h, lp, glp, initial_lp, F, β, γ, mb_size, μ, correctionΣ, preconΣ, std, Minv, 1, α, transf_mean, transf_std)
    end 
end 

function step!(st::State_θ, s::nonlinearBAOAB, stm::StatisticalModel) 
    K = nlinparams(stm)

    st.θ_prime -= 0.5 * s.h * s.F

    st.θ += 0.5s.h * [zeros(K, K) zeros(K, K); zeros(K, K) s.Minv] * st.θ_prime 

    st.θ_prime = exp(-s.h * s.γ) * st.θ_prime + sqrt((1/s.β) * (1 - exp(-2*s.γ*s.h))) * vcat(zeros(K), inv(sqrt.(s.Minv)) * randn(K))

    st.θ += 0.5s.h * [zeros(K, K) zeros(K, K); zeros(K, K) s.Minv] * st.θ_prime 

    s.F = s.glp(st.θ, [stm.data[i] for i in sample(1:length(stm.data), s.mb_size, replace = false)], length(stm.data))
    s.F[1:K] = zeros(K) 
    s.F[K+1:end] = s.F[K+1:end]

    st.θ_prime -= 0.5 * s.h * s.F 

    new_log_post_val = s.lp(st.θ, stm.data, length(stm.data)) 

    println(string(s.log_post_val) * " --n--> " * string(new_log_post_val) * " : " * string(new_log_post_val))

    s.log_post_val = new_log_post_val

    # update linear correction covariance matrix at every step 
    nonlin_comp_of_state = st.θ[K+1:end]
    s.correctionΣ = s.correctionΣ + (1/(s.t))*(( (nonlin_comp_of_state - s.μ) * transpose(nonlin_comp_of_state - s.μ)) - s.correctionΣ)
    
    # update linear mean 
    s.μ = s.μ + (1/(s.t)) * (nonlin_comp_of_state - s.μ)

    # update linear std dev matrix after 100 steps and every 20 steps. 
    if s.t > 100 && s.t % 10 == 0 
        estimated_Σ = s.α * s.preconΣ .+ (1 - s.α) * s.correctionΣ
        s.std = covariance_to_stddev(estimated_Σ)
        s.Minv = Diagonal(diag(estimated_Σ))    # mass = Precision ⟹ Inv Mass = Covariance
    end 

    s.t += 1 
end 

function run!(st::State_θ, s::Union{linearBAOAB, nonlinearBAOAB}, stm::StatisticalModel, Nsteps::Int64, outp) 
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

        push!(outp.mass, norm(s.Σ))
           
    end 
end 

end # end module 