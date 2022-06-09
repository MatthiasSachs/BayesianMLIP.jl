module Utils
using Plots 
using BayesianMLIP.Outputschedulers
using BayesianMLIP.NLModels

export animate, animation

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