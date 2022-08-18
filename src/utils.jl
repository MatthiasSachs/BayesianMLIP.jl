module Utils
using Plots, ACE
using Distributions
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.Outputschedulers
using BayesianMLIP.NLModels
using Flux, FluxOptTools, Zygote, ACEflux
import BayesianMLIP.NLModels: nparams 
using Distributed, ThreadsX

export StatisticalModel, params, nparams
export log_prior, log_likelihood, log_posterior
export get_glp, get_gll, get_glpr
export Histogram, Trajectory, Summary
export log_likelihood_parallel


function Histogram(outp) 
    i = [1, 2, 3, 4]
    true_vals = outp.θ[1] 

    l = @layout [a b ; c d]
    
    p1 = histogram([elem[i[1]] for elem in outp.θ], title="Index $(i[1]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=6)
    plot!([true_vals[i[1]]], seriestype="vline", color="red")

    p2 = histogram([elem[i[2]] for elem in outp.θ], title="Index $(i[2]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=6)
    plot!([true_vals[i[2]]], seriestype="vline", color="red")

    p3 = histogram([elem[i[3]] for elem in outp.θ], title="Index $(i[3]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=6)
    plot!([true_vals[i[3]]], seriestype="vline", color="red")

    p4 = histogram([elem[i[4]] for elem in outp.θ], title="Index $(i[4]) Trajectory", legend=false, titlefontsize=10, ytickfontsize=6)
    plot!([true_vals[i[4]]], seriestype="vline", color="red")

    plot(p1, p2, p3, p4, layout=l)

    # plot(O, layout=4, legend=false)
    # histogram(O, layout=4, legend=false)
end 

function Trajectory(outp) 
    len = length(outp.θ)

    i = [1, 2, 3, 4]
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

    plot(p1, p2, p3, p4, layout=l)
end 

function Summary(outp) 
    len = length(outp.θ)
    l = @layout [a b ; c ] 

    p1 = plot(1 .- outp.rejection_rate, title="Acceptance Rate", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)

    p2 = plot(outp.eigen_ratio, title="Condition Value", legend=false, titlefontsize=10,xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)

    p3 = plot(outp.log_posterior, title="Log-Posterior Values", legend=false, titlefontsize=10, xtick=false, xlabel="$len Steps", xguidefontsize=8, ytickfontsize=6)
    plot!([outp.log_posterior[1]], seriestype="hline", color="red")

    plot(p1, p2, p3, layout=l) 
end 



mutable struct StatisticalModel 
    log_likelihood
    prior
    pot
    data
end 

function Flux.params(pot::ACEflux.FluxPotential) 
    return Flux.params(pot.model)
end

function Flux.params(stm::StatisticalModel) 
    return Flux.params(stm.pot.model) 
end


function log_likelihood(stm::StatisticalModel)
    # log_likelihood for entire dataset 
    # return sum(stm.log_likelihood(stm.pot, d) for d in stm.data)
    # return sum(Distributed.pmap(d ->  stm.log_likelihood(stm.pot, d), stm.data; distributed=false))
    return sum(ThreadsX.map(d ->  stm.log_likelihood(stm.pot, d), stm.data))
end

function log_prior(stm::StatisticalModel)
    # logarithm of the prior 
    return logpdf(stm.prior, reshape(get_params(stm.pot), nparams(stm.pot))) 
end

# log posterior of Statistical model w/ current θ
log_posterior(stm::StatisticalModel) = log_likelihood(stm) #+ log_prior(stm)

function log_posterior(stm::StatisticalModel, θ)
    # log posterior of stm with chosen θ 
    set_params!(stm.pot, θ)
    return log_posterior(stm)
end

function get_gll(stm::StatisticalModel)
    # gradient of log likelihood wrt θ
    function gll(θ,d) 
        set_params!(stm.pot,θ)
        p = Flux.params(stm.pot)
        dL = Zygote.gradient(()->stm.log_likelihood(stm.pot, d), p)
        gradvec = zeros(p)
        copy!(gradvec,dL)
        return gradvec
    end
    return gll
end

function get_glpr(stm::StatisticalModel)
    # gradient of log prior wrt θ
    function glpr(θ::AbstractArray{T}) where {T<:Real} 
        return Zygote.gradient(θ->logpdf(stm.prior, θ), θ)[1]
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