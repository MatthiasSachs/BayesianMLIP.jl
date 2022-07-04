using ACEflux, JuLIP, ACE, Flux, Zygote
using ACE: State
using LinearAlgebra
using Printf
using Test, ACE.Testing
using ACEbase

using ACEflux: Linear_ACE
#model = Chain(Linear_ACE(2, 7, 4), Dense(4, 3, σ), Dense(3, 1), sum)
model = Chain(Linear_ACE(;ord = 2, maxdeg = 4,  Nprop=4), Dense(4, 3, σ), Dense(3, 1), sum)

pot = FluxPotential(model, 5.0) #model, cutoff

##

# # we only check the derivatives of the parameters in the linear ace layer
# # we do finite difference on the whole function, but only compare ACE parameters

@info "dEnergy, dE/dP"

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 

s = size(pot.model[1].weight)

function F(c)
   pot.model[1].weight = reshape(c, s[1], s[2])
   return energy(pot, at)
end

function dF(c)
   pot.model[1].weight = reshape(c, s[1], s[2])
   p = Flux.params(model)
   dE = Zygote.gradient(()->energy(pot, at), p)
   return(dE[p[1]])
end

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


