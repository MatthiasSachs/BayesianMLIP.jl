using ACEflux, JuLIP, ACE, Flux, Zygote
using ACEflux: Linear_ACE
using BayesianMLIP.Utils

#model = Chain(Linear_ACE(2, 7, 4), Dense(4, 3, σ), Dense(3, 1), sum)
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10


linear_ace_layer = Linear_ACE(;ord = 2, maxdeg = 4)
model = Chain(linear_ace_layer, GenLayer(FS), sum)
pot = FluxPotential(model, 6.0)

Σinv = get_precon(pot, .5, 2.0) # use a matrix proportional to Σ as a proposal in adMH