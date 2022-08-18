module Utils
using Plots, ACE
using Distributions
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.Outputschedulers
using BayesianMLIP.NLModels
using Flux, FluxOptTools, Zygote, ACEflux
import BayesianMLIP.NLModels: nparams 

export StatisticalModel, params, nparams
export log_prior, log_likelihood, log_posterior
export get_lp, get_ll, get_lpr, get_glp, get_gll, get_glpr
export plotTrajectory, histogramTrajectory, plotRejectionRate, plotEigenRatio, plotLogPosterior, plotξ, plotMomenta

function plotTrajectory(outp, index) 
    plot(1:length(outp.θ), [elem[index] for elem in outp.θ], title="Trajectory Index $index", legend=false)
end 

function histogramTrajectory(outp, index) 
    histogram(1:length(outp.θ), [elem[index] for elem in outp.θ], title="Histogram Index $index", bins= :scott)
end 

function plotMomenta(outp, index) 
    plot(1:length(outp.θ_prime), [elem[index] for elem in outp.θ_prime], title="Momenta Index $index", legend=false)
end 

function plotRejectionRate(outp) 
    plot(1:length(outp.rejection_rate), outp.rejection_rate, title="Rejection Rate", legend=false) 
end 

function plotEigenRatio(outp) 
    plot(1:length(outp.eigen_ratio), outp.eigen_ratio, title="Eigenvalue Ratio", legend=false)
end



mutable struct StatisticalModel{LL,PR,POT} 
    log_likelihood::LL
    prior::PR
    pot::POT
    data
end 

struct ConstantLikelihood end

struct FlatPrior end


function Flux.params(pot::ACEflux.FluxPotential) 
    return Flux.params(pot.model)
end

function Flux.params(stm::StatisticalModel) 
    return Flux.params(stm.pot.model) 
end



function get_ll(stm::StatisticalModel)
    function ll(θ,d) 
        set_params!(stm.pot,θ)
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
        return (total/length(batch)) * sum(ll(θ,d) for d in batch) + lpr(θ)
    end
    return lp
end



# function log_likelihood(stm::StatisticalModel)
#     # log_likelihood for entire dataset 
#     return (stm.log_likelihood === nothing ? 0.0 : sum(stm.log_likelihood(stm.pot, d) for d in stm.data))
# end

# function log_prior(stm::StatisticalModel)
#     # logarithm of the prior 
#     return  log(stm.prior(stm.pot))
# end

# # log posterior of Statistical model w/ current θ
# log_posterior(stm::StatisticalModel) = log_likelihood(stm) #+ log_prior(stm)

# function log_posterior(stm::StatisticalModel, θ)
#     # log posterior of stm with chosen θ 
#     set_params!(stm.pot, θ)
#     return log_posterior(stm)
# end

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



# Wrapper functions for FluxPotentials
nparams(pot::FluxPotential) = nparams(pot.model)
function nparams(model::Chain)
    return sum(length(p) for p in Flux.params(model))
end

params(pot::FluxPotential)  = Flux.params(pot.model)


function get_params(m)
    c = zeros(nparams(m))
    get_params!(c, m) 
end
get_params!(c::AbstractArray{T}, pot::FluxPotential)  where {T <: Real} = get_params!(c,pot.model)
get_params!(c::AbstractArray{T}, model::Chain)  where {T <: Real} = copy!(c,Flux.params(model))

set_params!(pot::FluxPotential, c::AbstractArray{T}) where {T <: Real} =  set_params!(pot.model, c) 

function set_params!(model::Chain, c::AbstractArray{T}) where {T <: Real}
    p = Flux.params(model)
    copy!(p,c) 
end


function get_precon(pot::FluxPotential, rel_scaling::T, p::T) where {T<:Real}
    # Only works for FS-type models (i.e., exactly only the first layer is of type Linear_ACE; and no other layers have parameters)
    linear_ace_layer = pot.model.layers[1]
    scaling = ACE.scaling(linear_ace_layer.m.basis,p)
    scaling[1] = 1.0
    return  Diagonal(cat(scaling, rel_scaling * scaling, dims=1))
end


# animation 

# function animation(outp::atoutp ; name::String="anim", trace=false)
#     anim = @animate for t in 1:length(outp.at_traj)
#         frame = outp.at_traj[t].X  
#         XYZ_Coords = [ [point[1] for point in frame], [point[2] for point in frame], [point[3] for point in frame] ]

#         if trace == true 
#             scatter!(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
#                     markercolor="black", legend=false)
#         else 
#             scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
#                     markercolor="black", legend=false)
#         end 
#     end
#     gif(anim, "$(name).mp4", fps=200)
# end


end 