module Utils
using Plots 
using Distributions
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.Outputschedulers
using BayesianMLIP.NLModels

using Flux, FluxOptTools, Zygote, ACEflux

export animation, StatisticalModel, log_posterior, gll, glp, log_likelihood, params


mutable struct StatisticalModel 
    log_likelihood
    prior
    model
    data
end 

function Flux.params(pot::ACEflux.FluxPotential) 
    return Flux.params(pot.model) 
end

function Flux.params(s::StatisticalModel) 
    return Flux.params(s.model) 
end

nparams(m::StatisticalModel) = nparams(m.model)

function log_likelihood(m::StatisticalModel)
    return sum(m.log_likelihood(m.model, d) for d in m.data)
end

# function log_prior(m::StatisticalModel)
#     return logpdf(m.prior, θ)
# end

log_posterior(m::StatisticalModel) = sum(m.log_likelihood(m.model, d) for d in m.data) #+ m.log_prior(Flux.params(m.model))

function log_posterior(m::StatisticalModel, θ) 
    set_params!(m.model, θ)
    return log_posterior(m)
end

function get_gll(sm::StatisticalModel)
    function gll(c,d) 
        set_params!(sm.model,c)
        p = Flux.params(sm.model)
        dL = Zygote.gradient(()->sm.log_likelihood(sm.model, d), p)
        gradvec = zeros(p)
        copy!(gradvec,dL)
        return(gradvec)
    end
    return gll
end

function get_glpr(sm::StatisticalModel)
    function glpr(c::AbstractArray{T}) where {T<:Real} 
        return Zygote.gradient(c->logpdf(sm.prior,c), c)[1]
    end
    return glpr
end

function get_glp(sm::StatisticalModel)
    gll, glpr =  get_gll(sm), get_glpr(sm)
    return get_glp(gll, glpr)
end

function get_glp(gll, glpr)
    function glp(c::AbstractArray, batch)
        return sum(gll(c,d) for d in batch) + glpr(c)
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

function animation(outp::atoutp ; name::String="anim", trace=false)
    anim = @animate for t in 1:length(outp.at_traj)
        frame = outp.at_traj[t].X  
        XYZ_Coords = [ [point[1] for point in frame], [point[2] for point in frame], [point[3] for point in frame] ]

        if trace == true 
            scatter!(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
                    markercolor="black", legend=false)
        else 
            scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
                    markercolor="black", legend=false)
        end 
    end
    gif(anim, "$(name).mp4", fps=200)
end


end 