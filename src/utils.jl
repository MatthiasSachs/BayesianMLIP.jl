module Utils

# animation 

function animate!(outp ; name::String="anim", trace=false)
    anim = @animate for t in 1:length(outp.X_traj)
        frame = outp.X_traj[t]  # a no_of_particles-vector with each element 
        XYZ_Coords = [ [point[1] for point in frame], [point[2] for point in frame], [point[3] for point in frame] ]

        if trace == true 
            scatter!(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
                    markercolor="black", legend=false)
        else 
            scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
                    markercolor="black", legend=false)
        end 
    end
    gif(anim, "$(name).mp4", fps=50)
end

end 