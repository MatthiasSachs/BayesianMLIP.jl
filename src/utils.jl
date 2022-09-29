module Utils
using Plots, ACE, NeighbourLists, Distributions, ACEflux, LinearAlgebra, Statistics
using ACEatoms: neighbourlist
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.Outputschedulers, BayesianMLIP
using BayesianMLIP.NLModels
using Flux, FluxOptTools, Zygote, ACEflux
import BayesianMLIP.NLModels: nparams, design_matrix, svecs2vec
import LinearAlgebra: eigen 
using ThreadsX

export StatisticalModel, params, nparams
export set_params!, get_params, get_params!
export get_lp, get_ll, get_lpr, get_glp, get_gll, get_glpr
export Histogram, Trajectory, Summary
export FlatPrior, ConstantLikelihood, get_precon, getmb 
export get_precon_params, get_Y, get_Σ_Tilde, precon_pre_cov_mean
export precision_to_covariance, precision_to_stddev, covariance_to_precision, covariance_to_stddev
export dampen, cnum


mutable struct StatisticalModel{LL,PR,POT} 
    log_likelihood::LL
    prior::PR
    pot::POT
    data
end 

function cnum(mat) 
    ev = eigen(mat).values 
    return maximum(ev)/minimum(ev)
end 

design_matrix(stm::StatisticalModel) = reduce(vcat, [design_matrix(stm.pot, stm.data[i].at) for i in 1:length(stm.data)])

function get_Y(stm::StatisticalModel) 
    egy = [d.E for d in stm.data]
    frc = [reduce(vcat, d.F) for d in stm.data]

    vec = [] 
    for i in 1:length(stm.data) 
        push!(vec, egy[i])
        push!(vec, frc[i])
    end 
    return reduce(vcat, vec)
end 

function get_Σ_Tilde(stm::StatisticalModel)
    at_lengths = [length(d.at) for d in stm.data]
    diag = reduce(vcat, [vcat([1], 3 * elem * ones(3 * elem)) for elem in at_lengths])
    return LinearAlgebra.Diagonal(diag)
end 

function precon_pre_cov_mean(stm::StatisticalModel) 
    # Calculates the closed-form precision matrix, covariance matrix, and true mean

    Ψ = design_matrix(stm)
    Y = get_Y(stm)
    Σ_0 = I
    Σ_Tilde = get_Σ_Tilde(stm)
    β = 1.0 

    # Compute precision matrix (stable) 
    cΣ = svd(Σ_Tilde); Σt_sqrt = Diagonal(sqrt.(cΣ.S)) * transpose(cΣ.U); Σt_sqrtΨ =  Σt_sqrt * Ψ
    precision_ = Σ_0 + β * transpose(Σt_sqrtΨ) * Σt_sqrtΨ

    # Compute true covariance using SVD
    # May be numerically unstable due to bad condition number
    Precision_svd = svd(precision_)
    covariance_sqrt = Precision_svd.U * Diagonal(1.0 ./ sqrt.(Precision_svd.S)) 
    Covariance = covariance_sqrt * transpose(covariance_sqrt)

    # Compute true mean
    μ_posterior = Vector{Float64}(β * Covariance * transpose(Ψ) * Y)

    dict = Dict{String, Any}("true_precision" => precision_, 
                             "true_covariance"  => Covariance, 
                             "true_mean" => μ_posterior) 

    return dict
end 

# precision to covariance is stable, but covariance to precision is not, even with SVD
# Cov = std * std^T
function precision_to_stddev(Σ_inv) 
    precision_svd = svd(Σ_inv) 
    return precision_svd.U * Diagonal(1.0 ./ sqrt.(precision_svd.S))
end

function precision_to_covariance(Σ_inv) 
    std = precision_to_stddev(Σ_inv)
    return std * transpose(std)
end 

function covariance_to_precision(Σ) 
    
end 

function covariance_to_stddev(Σ) 
    singValDec = svd(Σ) 
    return singValDec.U * Diagonal(sqrt.(singValDec.S))
end 

function dampen(matrix::Matrix{Float64}, c::Float64) 
    return matrix + c * I 
end 
    
function Histogram(outp::MHoutp_θ ; save_fig=false, title="") 
    i = [9, 10, 1, 1]
    true_vals = outp.θ[1] 
    l = @layout [a b ; c d]
    
    p1 = histogram([elem[i[1]] for elem in outp.θ], title="Index $(i[1]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1, bins=:scott)
    plot!([true_vals[i[1]]], seriestype="vline", color="red")

    p2 = histogram([elem[i[2]] for elem in outp.θ], title="Index $(i[2]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1)
    plot!([true_vals[i[2]]], seriestype="vline", color="red")

    p3 = histogram([elem[i[3]] for elem in outp.θ], title="Index $(i[3]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1)
    plot!([true_vals[i[3]]], seriestype="vline", color="red")

    p4 = histogram([elem[i[4]] for elem in outp.θ], title="Index $(i[4]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1)
    plot!([true_vals[i[4]]], seriestype="vline", color="red")

    if save_fig == true 
        plot(p1, p2, p3, p4, layout=l)
        savefig("./plots/$title.png")
    else 
        plot(p1, p2, p3, p4, layout=l)
    end 
end 

function Trajectory(outp::MHoutp_θ ; save_fig=false, title="") 
    len = length(outp.θ)
    i = [9, 10, 1, 1]
    true_vals = outp.θ[1] 
    l = @layout [a b ; c d]
    
    p1 = plot([elem[i[1]] for elem in outp.θ], title="Index $(i[1]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!([true_vals[i[1]]], seriestype="hline", color="red")

    p2 = plot([elem[i[2]] for elem in outp.θ], title="Index $(i[2]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!([true_vals[i[2]]], seriestype="hline", color="red")

    p3 = plot([elem[i[3]] for elem in outp.θ], title="Index $(i[3]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!([true_vals[i[3]]], seriestype="hline", color="red")

    p4 = plot([elem[i[4]] for elem in outp.θ], title="Index $(i[4]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!([true_vals[i[4]]], seriestype="hline", color="red")

    if save_fig == true 
        plot(p1, p2, p3, p4, layout=l)
        savefig("./plots/$title.png")
    else 
        plot(p1, p2, p3, p4, layout=l)
    end 
end 

function Summary(outp::MHoutp_θ ; save_fig=false, title="") 
    len = length(outp.θ)
    l = @layout [a b ; c d] 

    p1 = plot(1 .- outp.rejection_rate, title="Acceptance Rate", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)

    p2 = plot(outp.eigen_ratio, title="Condition Value", legend=false, titlefontsize=10,xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)

    p3 = plot(outp.log_posterior, title="Log-Posterior Values", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!([outp.log_posterior[1]], seriestype="hline", color="red")

    p4 = plot(outp.covariance_metric, title="Covariance Metric", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)

    

    if save_fig == true 
        plot(p1, p2, p3, p4, layout=l)
        savefig("./plots/$title.png")
    else 
        plot(p1, p2, p3, p4, layout=l)
    end 
      
end 


function get_precon_params(stm::StatisticalModel)
    basis = stm.pot.model[1].m.basis

    # outputs matrix of K × ∑_{d=1}^D N_d, where D is the number of data in stm.data
    # and N_d is the number of atomic environments in the 'd'th data

    mat = hcat([hcat(ThreadsX.map(i -> [B.val for B in ACE.evaluate(basis, [ ACE.State(rr = r)  for r in NeighbourLists.neigs(neighbourlist(d.at, stm.pot.cutoff), i)[2]] |> ACEConfig)], 1:length(d.at))...) for d in stm.data]...)

    std_dev = [std(mat[k, :]) for k in 1:length(basis)]

    avg = [mean(mat[k, :]) for k in 1:length(basis)]

    return hcat(avg, std_dev)
end 

struct ConstantLikelihood end

struct FlatPrior end



function get_ll(stm::StatisticalModel)
    function ll(θ,d) 
        BayesianMLIP.NLModels.set_params!(stm.pot,θ)
        return stm.log_likelihood(stm.pot, d)
    end
    return ll
end

function get_ll(stm::StatisticalModel{ConstantLikelihood,PR}) where {PR}
    return (θ,d)  -> 0.0
end

function get_lpr(stm::StatisticalModel) 
    function lpr(θ) 
        return logpdf(stm.prior, θ)
    end
    return lpr
end

function get_lpr(stm::StatisticalModel{LL,FlatPrior}) where {LL}
    return θ -> 0.0
end

function get_lp(stm::StatisticalModel)
    ll, lpr =  get_ll(stm), get_lpr(stm)
    return get_lp(ll, lpr)
end

function get_lp(ll, lpr)
    function lp(θ::AbstractArray, batch, total::Int64)
        return (total/length(batch)) * sum(ThreadsX.map(d -> ll(θ,d), batch)) + lpr(θ)
    end
    return lp
end


function get_gll(stm::StatisticalModel)
    # gradient of log likelihood wrt θ
    function gll(θ,d) 
        set_params!(stm.pot,θ)
        p = Flux.params(stm.pot)
        gradvec = zeros(p)
        dL = Zygote.gradient(()->stm.log_likelihood(stm.pot, d), p)
        copy!(gradvec,dL) 
        return gradvec
    end
    return gll
end


function get_gll(stm::StatisticalModel{ConstantLikelihood, PR}) where {PR}
    p = Flux.params(stm.pot)
    function gll(θ,d) 
        return zeros(p)
    end
    return gll
end

function get_glpr(stm::StatisticalModel{LL,PR}) where {LL, PR<:Distributions.Sampleable }
    # gradient of log prior wrt θ
    function glpr(θ::AbstractArray{T}) where {T<:Real} 
        return Zygote.gradient(θ->logpdf(stm.prior, θ), θ)[1]
    end
    return glpr
end

function get_glpr(stm::StatisticalModel{LL,FlatPrior}) where {LL}
    # gradient of log prior wrt θ
    p = Flux.params(stm.pot)
    function glpr(θ::AbstractArray{T}) where {T<:Real} 
        return zeros(p)
    end
    return glpr
end 


function get_glp(stm::StatisticalModel)
    gll, glpr =  get_gll(stm), get_glpr(stm)
    return get_glp(gll, glpr)
end

function get_glp(gll, glpr)
    function glp(θ::AbstractArray, batch, total::Int64)
        return (total/length(batch)) * sum(ThreadsX.map(d -> gll(θ,d), batch)) + glpr(θ)
    end
    return glp
end


# Get batch of size mbsize 

function getmb(data, mbsize) 
    return [data[i] for i in sample(1:length(data), mbsize, replace = false)]
end 

struct eigen_arr values end 

# eigen function for compatibility with UniformScaling 
function eigen(A)
    return eigen_arr([1, 1])
end 


function get_precon(pot::FluxPotential, rel_scaling::T, p::T) where {T<:Real}
    # Only works for FS-type models (i.e., exactly only the first layer is of type Linear_ACE; and no other layers have parameters)
    linear_ace_layer = pot.model.layers[1]
    scaling = ACE.scaling(linear_ace_layer.m.basis,p)
    scaling[1] = 1.0
    return  Diagonal([w*s for s in scaling for w in [1.0, rel_scaling]])
end



end 