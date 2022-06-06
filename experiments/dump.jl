function mainFinnisSinclairSimulation() 
    maxdeg = 4
    ord = 2
    Bsel = SimpleSparseBasis(ord, maxdeg)
    rcut = 5.0 
    
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
                                        rin = 1.2, rcut = 5.0)
    ACE.init1pspec!(B1p, Bsel)
    basis1 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
    basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
    
    
    at = bulk(:Ti, cubic=true) * 3
    
    rattle!(at,0.1) 
    model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x), 
                        x -> 1 / (2 * sqrt(x)), 
                        ones(length(basis1)), ones(length(basis2)))
    
    # E = energy(model, at)   
    F = forces(model, at)
    # grad1, grad2 = FS_paramGrad(model, at)
    
    VVIntegrator = VelocityVerlet(0.1, Vpair, at)
    PVIntegrator = PositionVerlet(0.1, Vpair, at)
    BAOIntegrator = BAOAB(0.1, Vpair, at)
    outp = simpleoutp()
    run!(VVIntegrator, model, at, 260; outp = outp)
    animate!(outp, name="FS_Animation")
end 

function mainMorseSimulation() 
    @info("Define (Morse) pair-potential")
    r0 = rnn(:Al)
    Vpair = JuLIP.morse(;A=4.0, e0=.5, r0=r0, rcut=(1.9*r0, 2.7*r0))
    
    @info("Create random Al configuration")
    seed!(1234)
    at = bulk(:Al, cubic=true) * 3
    at = rattle!(at, 0.1)
    
    # F = forces(Vpair, at)
    # E = energy(Vpair, at)
    
    VVIntegrator = VelocityVerlet(0.1, Vpair, at)
    PVIntegrator = PositionVerlet(0.1, Vpair, at)
    BAOIntegrator = BAOAB(0.1, Vpair, at; β=10.0)
    config_temp = []
    outp = simpleoutp()
    Nsteps = 1000
    run!(BAOIntegrator, Vpair, at, Nsteps; outp = outp, config_temp=config_temp)
    println(config_temp)
    println((1/Nsteps) * sum(config_temp))
    # animate!(outp, name="Morse_Animation")
end 

function config_temperature(F, X) 
    (1/(3 * length(F))) * sum([dot(-f, x) for (f, x) in zip(F, X)])
end

mainMorseSimulation()

function Hamiltonian(V, at::Atoms) 
    # Wish we would directly call this on outp, but this would require outp to store 
    # entire at object rather than at.X and at.P
    PE = energy(V, at)
    KE = 0.5 * sum([dot(at.P[t] /at.M[t], at.P[t]) for t in 1:length(at.P)])
    return PE + KE 
end 


# animation 

# Move to utils.jl
# function animate!(outp ; name::String="anim", trace=false)
#     anim = @animate for t in 1:length(outp.X_traj)
#         frame = outp.X_traj[t]  # a no_of_particles-vector with each element 
#         XYZ_Coords = [ [point[1] for point in frame], [point[2] for point in frame], [point[3] for point in frame] ]

#         if trace == true 
#             scatter!(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
#                     markercolor="black", legend=false)
#         else 
#             scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3], title="Trajectory", framestyle=:grid, marker=2, 
#                     markercolor="black", legend=false)
#         end 
#     end
#     gif(anim, "$(name).mp4", fps=50)
# end