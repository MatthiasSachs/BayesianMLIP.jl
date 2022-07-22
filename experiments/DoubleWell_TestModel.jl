using Plots
using ACE, ACEatoms, JuLIP, Plots, ACEflux, Flux, Zygote, LinearAlgebra, JLD2
using BayesianMLIP           
using BayesianMLIP.NLModels         
using BayesianMLIP.Dynamics
using BayesianMLIP.Outputschedulers
using BayesianMLIP.Utils
using BayesianMLIP.MHoutputschedulers

U(q) = 3(q^2 - 1)^2 + q + 1
F(q) = -12q^3 + 12q - 1
plot(U, -2, 2)
Gibbs(q) = exp(-U(q))
plot!(Gibbs, -2, 2)



mutable struct mState  
    q::Float64 
    p::Float64 
end 

mutable struct mBAOAB 
    h::Float64      # Step size
    β::Float64 
    γ::Float64 
end 

function mstep!(st::mState, s::mBAOAB) 
    st.p += 0.5 * s.h * F(st.q) 
    st.q += 0.5 * s.h * st.p 
    st.p = exp(-s.h * s.γ) * st.p + sqrt((1/s.β) * (1 - exp(-2*s.γ*s.h))) * randn() 
    st.q += 0.5 * s.h * st.p 
    st.p += 0.5 * s.h * F(st.q)
end 

function mrun!(st::mState, sampler::mBAOAB, Nsteps::Int64, outp) 
    push!(outp[1], st.q)
    push!(outp[2], st.p)
    for _ in 1:Nsteps 
        mstep!(st, sampler) 
        push!(outp[1], st.q)
        push!(outp[2], st.p)
    end 
end 


outp = [[], []]
init = mState(0.0, 0.0) 
sampler = mBAOAB(0.01, 1.0, 1.0)
mrun!(init, sampler, 100000, outp)

histogram(outp[1], bins = :scott)
plot(1:length(outp[1]), outp[1])
plot(1:length(outp[1]), U.(outp[1]))
