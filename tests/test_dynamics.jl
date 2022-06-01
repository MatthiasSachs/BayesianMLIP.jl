using JuLIP, Random
using Plots 
using ACE

@info("Define (Morse) pair-potential")
r0 = rnn(:Al)
Vpair = JuLIP.morse(;A=4.0, e0=.5, r0=r0, rcut=(1.9*r0, 2.7*r0))

@info("Create random Al configuration")
Random.seed!(1234)
at = bulk(:Al, cubic=true) * 3
at = rattle!(at, 0.1)

# XYZ_Coords = [ [point[1] for point in at.X], [point[2] for point in at.X], [point[3] for point in at.X] ]
# scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3])     # Quick inspection of initial positions of particles


# Compute forces 
F = forces(Vpair,at)
# Compute energy
E = energy(Vpair,at)

using JuLIP: AbstractAtoms 

abstract type Dynamics end
abstract type HamiltonianDynamics <: Dynamics end


mutable struct VelocityVerlet{T} <: HamiltonianDynamics where {T}
    F::Vector{JVec{T}} # force
    h::Float64      # step size
end

mutable struct PositionVerlet{T} <: HamiltonianDynamics where {T}
    F::Vector{JVec{T}} # force
    h::Float64      # step size
end

B_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_momenta!(at, at.P + hf * d.h * d.F)
A_step!(d::HamiltonianDynamics, at::AbstractAtoms; hf::T=1.0) where {T<:Real} = set_positions!(at, at.X + hf * d.h * at.P./at.M)

function step!(s::VelocityVerlet, V, at::AbstractAtoms ) #V::SitePotential (e.g. FSModel)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=1.0)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end

function step!(s::PositionVerlet, V, at::AbstractAtoms ) #V::SitePotential
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=1.0)
    A_step!(s, at; hf=.5)
end

# animation 
function animate!(d::Dynamics, V, N::Int)
    anim = @animate for _ in 1:N
        try 
            step!(d, V, at)
            println(energy(V, at) + 0.5 * transpose(at.P./at.M) * at.P )
            XYZ_Coords = [ [point[1] for point in at.X], [point[2] for point in at.X], [point[3] for point in at.X] ]
            scatter(XYZ_Coords[1], XYZ_Coords[2], XYZ_Coords[3])
        catch e
            print("Error happened")
            break 
        end
    end

    gif(anim, "anim.mp4", fps=100)
end


VVObj = VelocityVerlet(F, 0.05)
step!(VVObj, Vpair, at)
animate!(VVObj, Vpair, 200)

PVObj = PositionVerlet(F, 0.05) 
step!(PVObj, Vpair, at)
animate!(PVObj, Vpair, 500)

abstract type GradientDynamics <: Dynamics end

abstract type OLDDynamics <: GradientDynamics end

mutable struct EulerMaruyama{T} <: OLDDynamics where {T<:Real}
    h::T 
    β::T    # step size
 end

function step!(d::EulerMaruyama, V, at::AbstractAtoms) where {T}
    set_positions!(at, at.X + 1.0 * d.h * forces(V, at)./at.M + sqrt.( 1.0/d.β * d.h./at.M).*randn(ACE.SVector{3,Float64},length(at)))
end

EMObj = EulerMaruyama(0.01, 1.0) 
step!(EMObj, Vpair, at)
animate!(EMObj, Vpair, 200)

abstract type Thermostat <: HamiltonianDynamics end

get_invtemp(d::Thermostat) = d.β

abstract type Langevin <: Thermostat end 

O_step!(s::Langevin, at::AbstractAtoms) = set_momenta!(at, s.α .* at.P + s.ζ * randn(ACE.SVector{3,Float64},length(at)) )

mutable struct BAOAB{T} <: Langevin where {T<:Real}  
    h::T        # Step size 
    F::Vector{ACE.SVector{3,T}}
    β::T        # Inverse Temperature 
    α::T        # Integrator parameters
    ζ::T        # Integrator parameters 

end

# BAOAB(h::T, N::Int; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOAB(h, zeros(ACE.SVector{3,T},N), β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ))))  

# Constructor for BAOAB struct 
BAOAB(h::T, F::Vector{ACE.SVector{3,T}}; γ::T=1.0, β::T=1.0) where {T<:Real} = BAOAB(h, F, β, exp(-h *γ ), sqrt( 1.0/β * (1-exp(-2*h*γ))))

function step!(s::BAOAB, V, at::AbstractAtoms)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
    A_step!(s, at; hf=.5)
    O_step!(s, at)
    A_step!(s, at; hf=.5)
    s.F = forces(V, at)
    B_step!(s, at; hf=.5)
end

BAObj = BAOAB(0.01, F; β = 2.0)
step!(BAObj, Vpair, at) 
animate!(BAObj, Vpair, 100)


println(at)
at.pbc = (false, false, false)
sum(sum(-f .* x for (f, x) in zip(F[t], at.X[t])) for t in 1:100)/100
sum(sum(-f .* x for (f, x) in zip(F[t], at.X[t]))/(3*3) for t in 1:54)/54