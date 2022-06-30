module Utils
using Plots 
using Distributions
using Distributions: logpdf, MvNormal
using LinearAlgebra 
using BayesianMLIP.Outputschedulers
using BayesianMLIP.NLModels

export animation, StatisticalModel, log_posterior

mutable struct StatisticalModel 
    log_likelihood
    prior
    model
    data
end 

function log_posterior(m::StatisticalModel, θ) 
    set_params!(m.model, θ)
    return sum(m.log_likelihood(m.model, d) for d in m.data) + logpdf(m.prior, θ)
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


# using Plots
# x=-1.0:1.0:1.0
# y=-10.0:1.0:10.0

# function log_post(x::Float64, y::Float64) 
#     test = true_θ 
#     test[1] = x 
#     test[2] = y
#     return U(statModel, test)
# end 

# surface(x,y,log_post)

end 