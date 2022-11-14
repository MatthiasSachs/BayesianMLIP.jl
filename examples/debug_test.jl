using ACE, ACEatoms, ACEflux, Flux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics, JSON, Plots, BenchmarkTools
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Utils, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser, BayesianMLIP.globalSamplers, BayesianMLIP.conditionalSamplers
import ACEflux: FluxPotential 
import Distributions: MvNormal

x = randn(1000)

x = 1

zero(Int64)

function g() 
    y = 0 
    for _ in 1:1_000_000 
        y += x
    end 
end 

@btime g()




function f(x)  
    s = zero(eltype(x))
    for val in x 
        s = s + val 
    end 
    return s 
end 

@btime f($x)


@code_warntype f(randn(1000))

