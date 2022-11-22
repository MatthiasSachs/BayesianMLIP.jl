module Utils
using Plots, ACE, NeighbourLists, Distributions, ACEflux, LinearAlgebra, Statistics, ThreadsX, Flux, FluxOptTools, Zygote, JuLIP
using ACE: LinearACEModel
using ACEatoms: neighbourlist
using Distributions: logpdf, MvNormal
using BayesianMLIP.Outputschedulers, BayesianMLIP, BayesianMLIP.NLModels
import BayesianMLIP.NLModels: nparams, nlinparams, svecs2vec

export StatisticalModel, params, nparams
export set_params!, get_params, get_params!
export get_lp, get_ll, get_lpr, get_glp, get_gll, get_glpr
export Histogram, Trajectory, Summary
export FlatPrior, ConstantLikelihood, get_precon, getmb 
export preconChangeOfBasis, cnum
export precision_to_covariance, precision_to_stddev, covariance_to_precision, covariance_to_stddev
export get_Σ_Tilde, get_Y, design_matrix


mutable struct StatisticalModel{LL,PR,POT} 
    log_likelihood::LL
    prior::PR
    pot::POT
    data
end 

nparams(stm::StatisticalModel) = nparams(stm.pot)

nlinparams(stm::StatisticalModel) = nlinparams(stm.pot)

function cnum(mat::Matrix{Float64}) 
    ev = LinearAlgebra.eigen(mat).values 
    return maximum(ev) / minimum(ev)
end 

function design_matrix(model::LinearACEModel, at::AbstractAtoms) 
    # Finds the design matrix 
    bsis_energy = transpose(basis_energy(model, at))
    bsis_forces = reduce(hcat, svecs2vec.(basis_forces(model, at)))
    return vcat(bsis_energy, bsis_forces)
end 

function design_matrix(pot::FluxPotential, at::AbstractAtoms) 
    # Finds the design matrix 
    bsis_energy = transpose(basis_energy(pot, at))
    bsis_forces = reduce(hcat, svecs2vec.(basis_forces(pot, at)))
    return vcat(bsis_energy, bsis_forces)
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
    return diag
end 

function preconChangeOfBasis(stm::StatisticalModel) 
    # Calculates the closed-form precision matrix, covariance matrix, and true mean

    Ψ = design_matrix(stm)
    Y = get_Y(stm)
    Σ_0 = I 
    Σ_Tilde = get_Σ_Tilde(stm)
    β = 1.0 

    H = Diagonal(sqrt.(Σ_Tilde)) * Ψ
    lin_precision = Σ_0 + β * (transpose(H) * H)

    svdH = svd(H) 
    
    lin_covariance = svdH.V * Diagonal(1 ./ (1 .+ svdH.S.^2)) * svdH.Vt 
    nlin_covariance = svdH.V * Diagonal((1 ./ (1 .+ svdH.S.^2).^2)) * svdH.Vt 
    lin_std = svdH.V * Diagonal(sqrt.(1 ./ (1 .+ svdH.S.^2))) 
    nlin_std = svdH.V * Diagonal(1 ./ (1 .+ svdH.S.^2))

    # Compute true mean
    μ_posterior = Vector{Float64}(β * lin_covariance * transpose(Ψ) * Y)

    dict = Dict{String, Any}("lin_covariance" => lin_covariance, 
                             "lin_precision" => lin_precision, 
                             "lin_std" => lin_std, 
                             "lin_mean" => μ_posterior, 
                             "nlin_covariance" => nlin_covariance, 
                             "nlin_std" => nlin_std, 
                             "nlin_mean" => zeros(length(μ_posterior))) 

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

function covariance_to_stddev(Σ) 
    singValDec = svd(Σ) 
    return singValDec.U * Diagonal(sqrt.(singValDec.S))
end 

    
function Histogram(outp::outp, i = [1, 2, 3, 4, 5, 6]; save_fig=false, title="") 
    true_vals = outp.θ[1] 
    l = @layout [a b c; d e f]
    
    p1 = histogram([elem[i[1]] for elem in outp.θ], title="Index $(i[1]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1, xtickfontsize=5, xrotation=30)
    plot!([true_vals[i[1]]], seriestype="vline", color="red")

    p2 = histogram([elem[i[2]] for elem in outp.θ], title="Index $(i[2]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1, xtickfontsize=5, xrotation=30)
    plot!([true_vals[i[2]]], seriestype="vline", color="red")

    p3 = histogram([elem[i[3]] for elem in outp.θ], title="Index $(i[3]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1, xtickfontsize=5, xrotation=30)
    plot!([true_vals[i[3]]], seriestype="vline", color="red")

    p4 = histogram([elem[i[4]] for elem in outp.θ], title="Index $(i[4]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1, xtickfontsize=5, xrotation=30)
    plot!([true_vals[i[4]]], seriestype="vline", color="red")
    
    p5 = histogram([elem[i[5]] for elem in outp.θ], title="Index $(i[5]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1, xtickfontsize=5, xrotation=30)
    plot!([true_vals[i[5]]], seriestype="vline", color="red")

    p6 = histogram([elem[i[6]] for elem in outp.θ], title="Index $(i[6]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=1, xtickfontsize=5, xrotation=30)
    plot!([true_vals[i[6]]], seriestype="vline", color="red")

    if save_fig == true 
        plot(p1, p2, p3, p4, p5, p6, layout=l)
        savefig("./plots/$title.png")
    else 
        plot(p1, p2, p3, p4, p5, p6, layout=l)
    end 
end 

function Trajectory(outp::outp, i = [1, 2, 3, 4, 5, 6]; save_fig=false, title="") 
    len = length(outp.θ)
    true_vals = outp.θ[1] 
    l = @layout [a b c; d e f]
    
    p1 = plot([elem[i[1]] for elem in outp.θ], title="Index $(i[1]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=5, yrotation=45)
    plot!([true_vals[i[1]]], seriestype="hline", color="red")

    p2 = plot([elem[i[2]] for elem in outp.θ], title="Index $(i[2]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=5, yrotation=45)
    plot!([true_vals[i[2]]], seriestype="hline", color="red")

    p3 = plot([elem[i[3]] for elem in outp.θ], title="Index $(i[3]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=5, yrotation=45)
    plot!([true_vals[i[3]]], seriestype="hline", color="red")

    p4 = plot([elem[i[4]] for elem in outp.θ], title="Index $(i[4]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=5, yrotation=45)
    plot!([true_vals[i[4]]], seriestype="hline", color="red")

    p5 = plot([elem[i[5]] for elem in outp.θ], title="Index $(i[5]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=5, yrotation=45)
    plot!([true_vals[i[5]]], seriestype="hline", color="red")

    p6 = plot([elem[i[6]] for elem in outp.θ], title="Index $(i[6]) Trajectory", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=5, yrotation=45)
    plot!([true_vals[i[6]]], seriestype="hline", color="red")

    if save_fig == true 
        plot(p1, p2, p3, p4, p5, p6, layout=l)
        savefig("./plots/$title.png")
    else 
        plot(p1, p2, p3, p4, p5, p6, layout=l)
    end 
end 

function Summary(outp::outp ; save_fig=false, title="") 
    len = length(outp.θ)
    l = @layout [a b ; c d] 

    p1 = plot(outp.acceptance_rate, title="Acceptance Rate", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!(outp.acceptance_rate_lin, legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6, color="green")
    plot!(outp.acceptance_rate_nlin, legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6, color="red")

    p2 = plot(outp.eigen_ratio, title="Covariance Info", label="Condition #", legend=:topleft, titlefontsize=10,xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6, color="blue")
    plot!(outp.covariance_metric, label="Norm", legend=:topleft, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6, color="red")

    p3 = plot(outp.log_posterior, title="Log-Posterior Values", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!([outp.log_posterior[1]], seriestype="hline", color="red")

    p4 = plot(outp.mass, title="Mass Matrix Info", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)

    # plot!(outp.F1, legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6, color="blue")
    # plot!(outp.F2, legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6, color="red")
    # plot!(outp.F3, legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6, color="green")
    

    if save_fig == true 
        plot(p1, p2, p3, p4, layout=l)
        savefig("./plots/$title.png")
    else 
        plot(p1, p2, p3, p4, layout=l)
    end 
      
end 


# function get_precon_params(stm::StatisticalModel)
#     basis = stm.pot.model[1].m.basis

#     # outputs matrix of K × ∑_{d=1}^D N_d, where D is the number of data in stm.data
#     # and N_d is the number of atomic environments in the 'd'th data

#     mat = hcat([hcat(ThreadsX.map(i -> [B.val for B in ACE.evaluate(basis, [ ACE.State(rr = r)  for r in NeighbourLists.neigs(neighbourlist(d.at, stm.pot.cutoff), i)[2]] |> ACEConfig)], 1:length(d.at))...) for d in stm.data]...)

#     std_dev = [std(mat[k, :]) for k in 1:length(basis)]

#     avg = [mean(mat[k, :]) for k in 1:length(basis)]

#     return hcat(avg, std_dev)
# end 

struct ConstantLikelihood end

struct FlatPrior end

function get_ll(stm::StatisticalModel, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm))))
    function ll(θ,d) 
        BayesianMLIP.NLModels.set_params!(stm.pot, transf_std * θ + transf_μ)
        return stm.log_likelihood(stm.pot, d)
    end
    return ll
end

function get_ll(stm::StatisticalModel{ConstantLikelihood,PR}, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm)))) where {PR}
    return (θ,d)  -> 0.0
end

function get_lpr(stm::StatisticalModel, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm)))) 
    function lpr(θ) 
        return logpdf(stm.prior, transf_std * θ + transf_μ)
    end
    return lpr
end

function get_lpr(stm::StatisticalModel{LL,FlatPrior}, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm)))) where {LL}
    return θ -> 0.0
end

function get_lp(stm::StatisticalModel, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm))))
    twoK = nparams(stm) 
    @assert length(transf_μ) == twoK
    @assert size(transf_std) == (twoK, twoK)
    ll, lpr =  get_ll(stm, transf_μ, transf_std), get_lpr(stm, transf_μ, transf_std)
    return get_lp(ll, lpr)
end

function get_lp(ll, lpr)
    function lp(θ::AbstractArray, batch, total::Int64)
        return (total/length(batch)) * sum(ThreadsX.map(d -> ll(θ,d), batch)) + lpr(θ)
    end
    return lp
end


function get_gll(stm::StatisticalModel, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm))))
    # gradient of log likelihood wrt θ
    
    function gll(θ,d) 
        set_params!(stm.pot, transf_std * θ + transf_μ)
        p = Flux.params(stm.pot.model)
        dL = Zygote.gradient(()->stm.log_likelihood(stm.pot, d), p)
        gradvec = zeros(p)
        copy!(gradvec, dL) 
        return - transf_std * vcat(gradvec[1:2:end], gradvec[2:2:end]) 
    end
    return gll
end

function get_gll(stm::StatisticalModel{ConstantLikelihood, PR}, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm)))) where {PR}
    function gll(θ,d) 
        return zeros(nparams(stm))
    end
    return gll
end

function get_glpr(stm::StatisticalModel{LL,PR}, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm)))) where {LL, PR<:Distributions.Sampleable}
    # gradient of log prior wrt θ
    function glpr(θ::AbstractArray{T}) where {T<:Real} 
        grad = Zygote.gradient(θ->logpdf(stm.prior, transf_std * θ + transf_μ), θ)[1] 
        return - grad 
    end
    return glpr
end

function get_glpr(stm::StatisticalModel{LL,FlatPrior}, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm)))) where {LL}
    # gradient of log prior wrt θ
    function glpr(θ::AbstractArray{T}) where {T<:Real} 
        return zeros(nparams(stm))
    end
    return glpr
end 


function get_glp(stm::StatisticalModel, transf_μ::Vector{Float64}=zeros(nparams(stm)), transf_std::Union{Matrix, Diagonal}=Diagonal(ones(nparams(stm))))
    twoK = nparams(stm) 
    @assert length(transf_μ) == twoK
    @assert size(transf_std) == (twoK, twoK)
    gll, glpr = get_gll(stm, transf_μ, transf_std), get_glpr(stm, transf_μ, transf_std)
    return get_glp(gll, glpr)
end

function get_glp(gll, glpr)
    function glp(θ::AbstractArray, batch, total::Int64)
        return (total/length(batch)) * sum(gll(θ, d) for d in batch) + glpr(θ)
        # return (total/length(batch)) * sum(ThreadsX.map(d -> gll(θ, d), batch)) + glpr(θ)
    end
    return glp
end


# Get batch of size mbsize 

function getmb(data, mbsize) 
    return [data[i] for i in sample(1:length(data), mbsize, replace = false)]
end 

# function get_precon(pot::FluxPotential, rel_scaling::T, p::T) where {T<:Real}
#     # Only works for FS-type models (i.e., exactly only the first layer is of type Linear_ACE; and no other layers have parameters)
#     linear_ace_layer = pot.model.layers[1]
#     scaling = ACE.scaling(linear_ace_layer.m.basis,p)
#     scaling[1] = 1.0
#     return  Diagonal([w*s for s in scaling for w in [1.0, rel_scaling]])
# end



end 