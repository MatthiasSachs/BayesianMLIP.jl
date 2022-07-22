using ACEflux, JuLIP, ACE, Flux, Zygote
using ACE: State
using LinearAlgebra
using Printf
using Test, ACE.Testing
using ACEbase
using StaticArrays

using ACEflux: Linear_ACE, GenLayer
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

model = Chain(Linear_ACE(;ord = 2, maxdeg = 4, Nprop = 2), GenLayer(FS), sum)
# fieldnames(typeof(model[1]))
# model[1].weight
# model[1].m
pot = FluxPotential(model, 6.0) 



##

# # we only check the derivatives of the parameters in the linear ace layer
# # we do finite difference on the whole function, but only compare ACE parameters

@info "dEnergy, dE/dP"

at = bulk(:Cu, cubic=true) * 3;
rattle!(at,0.6) ;

s = size(pot.model[1].weight)

energy(pot, at)      # much faster 
forces(pot, at)      # much faster 

function F(c)     # c is vector of length 30 
   pot.model[1].weight = reshape(c, s[1], s[2])
   return energy(pot, at)
end

function dF(c)
   pot.model[1].weight = reshape(c, s[1], s[2])
   p = Flux.params(model)  
   dE = Zygote.gradient(()->energy(pot, at), p)
   return dE[p[1]]
end

F(rand(30))
dF(rand(30))

for _ in 1:1
   c = rand(s[1]*s[2])
   println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
end
println()

##

@info "dForces, d{sum(F)}/dP"

sqr(x) = x.^2
ffrcs(pot, at) = sum(sum(sqr.(forces(pot, at))))

function F(c)
   pot.model[1].weight = reshape(c, s[1], s[2])
   return ffrcs(pot, at)
end

function dF(c)
   pot.model[1].weight = reshape(c, s[1], s[2])
   p = Flux.params(model)
   dF = Zygote.gradient(() -> ffrcs(pot, at), p)
   return(dF[p[1]])
end

for _ in 1:1
   c = rand(s[1]*s[2])
   println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
end
println()

##

@info "dloss, d{E+sum(F)}/dP"

loss(pot, at) = energy(pot, at) + sum(sum(sqr.(forces(pot, at))))

function F2(c)
   pot.model[1].weight = reshape(c, s[1], s[2])
   return loss(pot, at)
end

function dF2(c)
   pot.model[1].weight = reshape(c, s[1], s[2])
   p = Flux.params(model)
   dL = Zygote.gradient(()->loss(pot, at), p)
   return(dL[p[1]])
end

for _ in 1:1
   c = rand(s[1]*s[2])
   println(@test ACEbase.Testing.fdtest(F2, dF2, c, verbose=true))
end
println()


