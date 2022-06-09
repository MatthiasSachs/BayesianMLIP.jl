using BayesianMLIP.NLModels
using BayesianMLIP.Dynamics
using BayesianMLIP.Outputschedulers
using BayesianMLIP.Utils
using ACE, JuLIP
using LinearAlgebra: norm
 
# 1) Check that force works properly for linear version of fsmodel         
# 2) Check regularized non-linear version ( + 0.1, sigmoid?, arctan?)
# 3) Create synthetic data from a linear model with order = 3 or higher
# 4) Use non-linear FSmodel with order = 2 to learn same potential       
# 5) E_i  = energy(model,at_i) + \epsilon_i (1-dimensional), F_i = force(model,at_i) + \epsilon'_i (3N-dimensional), i = 1, ... N_obs
# 6) Use BAOAB on the logLikelihood function as the potential, with \beta very large, or \beta = log(t) "cooling schedule"
#    If beta is very large (i.e. cool temperature), then the integrator should converge to a minimum. When the temperature is cool, the Gibbs measure should be more concentrated, and running BAOAB will allow it to converge to global minima. 
# 7) Can also implement gradient descent if possible 

function demonstrate1() 
    # 1) Check that force works properly for linear version of fsmodel with coefficient of ones
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
    rattle!(at, 0.1)
     
    model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x+0.01),         # had to include the +0.01 since otherwise we would get 0*inf = NaN 
                        x -> 1 / (2 * sqrt(x+0.01)), 
                        ones(length(basis1)), zeros(length(basis2)))

    outp = atoutp()     # contains data on at, energy, and forces for each step 

    sampler = BAOAB(0.01, model, at) 
    nsteps = 5000
    run!(sampler, model, at, nsteps; outp=outp)
    animation(outp)
    for step in 1:nsteps 
        println(outp.at_traj[step].X[1])
    end

    # Results: Unstable (explodes) within 5000 steps 
end 

function demonstrate2()
    # 2) Check that system is stable for linear FS model with Uniform(0, 1) coefficients 
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
    rattle!(at, 0.1)
     
    model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x+0.01),         # had to include the +0.01 since otherwise we would get 0*inf = NaN 
                        x -> 1 / (2 * sqrt(x+0.01)), 
                        rand(length(basis1)), zeros(length(basis2)))

    outp = atoutp()     # contains data on at, energy, and forces for each step 

    sampler = BAOAB(0.01, model, at) 
    nsteps = 5000
    run!(sampler, model, at, nsteps; outp=outp)
    animation(outp)
    for step in 1:nsteps 
        println(outp.at_traj[step].X[1])
    end

    # Results: Stable within 5000 steps 
end 

function demonstrate3() 
    # 3) Check that system is stable with nonlinear FS model with Uniform(0, 1) coefficients
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
    rattle!(at, 0.1)
     
    model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x+0.01),         # had to include the +0.01 since otherwise we would get 0*inf = NaN 
                        x -> 1 / (2 * sqrt(x+0.01)), 
                        rand(length(basis1)), rand(length(basis2)))

    outp = atoutp()     # contains data on at, energy, and forces for each step 

    sampler = BAOAB(0.01, model, at) 
    nsteps = 5000
    run!(sampler, model, at, nsteps; outp=outp)
    animation(outp)
    for step in 1:nsteps 
        println(outp.at_traj[step].X[1])
    end

    # Results: Explodes within 5000 steps, Stable within 5000 steps in second try
    # DomainError in third try
end 

function demonstrate4() 
    # Check stability of randomized (coef) nonlinear FS model with sigmoid function. 
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
    rattle!(at, 0.1)
     
    model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> (1.0 / (1.0 + exp(-x))) - 0.5,         # had to include the +0.01 since otherwise we would get 0*inf = NaN 
                        x -> exp(-x) / ((1 + exp(-x))^2), 
                        rand(length(basis1)), rand(length(basis2)))

    outp = atoutp()     # contains data on at, energy, and forces for each step 

    sampler = BAOAB(0.01, model, at) 
    nsteps = 5000
    run!(sampler, model, at, nsteps; outp=outp)
    animation(outp)
    for step in 1:nsteps 
        println(outp.at_traj[step].X[1])
    end

    # Result: Unstable, explodes up to 50000
end 

function demonstrate5() 
    # Create data for linear model of order 3 
    maxdeg = 4
    ord = 3
    Bsel = SimpleSparseBasis(ord, maxdeg)
    rcut = 5.0 
    
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
                                        rin = 1.2, rcut = 5.0)
    ACE.init1pspec!(B1p, Bsel)
    basis1 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
    basis2 = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
    
    
    at = bulk(:Ti, cubic=true) * 3
    rattle!(at, 0.1)
     
    model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x+0.01),         # had to include the +0.01 since otherwise we would get 0*inf = NaN 
                        x -> 1 / (2 * sqrt(x+0.01)), 
                        rand(length(basis1)), zeros(length(basis2)))

    outp = atoutp()     # contains data on at, energy, and forces for each step 

    sampler = BAOAB(0.01, model, at) 
    nsteps = 5000
    run!(sampler, model, at, nsteps; outp=outp)
    animation(outp)
    for step in 1:nsteps 
        println(outp.at_traj[step].X[1])
    end

    # Results: Stable 
end 

# Collect data for linear system of order 2
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
rattle!(at, 0.1) 
model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x+0.01), 
                        x -> 1 / (2 * sqrt(x+0.01)), 
                        rand(length(basis1)), zeros(length(basis2)))

r0 = rnn(:Al) 
Vpair = JuLIP.morse(;A=4.0, e0=.5, r0=r0, rcut=(1.9*r0, rcut)) 


N_obs = 10
nsteps = 1000
sampler = BAOAB(0.01, model, at; γ=1.0, β=1.0)
data = []
for _ in 1:N_obs        # Take approx 500 secs for 100 iterations 
    run!(sampler, model, at, nsteps; outp = nothing)
    push!(data, (at = deepcopy(at), E= energy(model,at), F = forces(model,at) ))
end

for elem in data 
    println(elem.at.X[1])
end

# Problems: The energy and forces are all 0? 

struct GaussianNoiseLL
    model::FSModel 
    weight      # energy variance vs force variance weight ratio (should be of order 1/(3 N_atoms)) 
    data
end

model = FSModel(basis1, basis2, 
                        rcut, 
                        x -> -sqrt(x+0.01), 
                        x -> 1 / (2 * sqrt(x+0.01)), 
                        rand(length(basis1)), zeros(length(basis2)))

sm = GaussianNoiseLL(model, 1/54, data)

function Likelihood(sm::GaussianNoiseLL)
    # sm.model.c1 = c1 
    # sm.model.c2 = c2
    N_obs = length(sm.data)
    cost = 0.0
    for i in 1:N_obs 
        energy_diff = abs(sm.data[i].E - energy(sm.model, sm.data[i].at))^2
        force_diff = norm(sm.data[i].F - forces(sm.model, sm.data[i].at))^2 
        cost += -(energy_diff + sm.weight * force_diff)
    end 
    
    return cost
    # - sum_{i=1}^{N_obs} \Big( || E_i - energy(model,at_i)||^2 + weight || F_i - forces(model,at_i)||^2 \Big) 
end

Likelihood(sm)

function logLikelihood_d(sm::GaussianNoiseLL, c1, c2)

end


# 4) Use non-linear FSmodel with order = 2 to learn same potential       
# 5) E_i  = energy(model,at_i) + \epsilon_i (1-dimensional), F_i = force(model,at_i) + \epsilon'_i (3N-dimensional), i = 1, ... N_obs
# 6) Use BAOAB on the logLikelihood function as the potential, with \beta very large, or \beta = log(t) "cooling schedule"
#    If beta is very large (i.e. cool temperature), then the integrator should converge to a minimum. When the temperature is cool, the Gibbs measure should be more concentrated, and running BAOAB will allow it to converge to global minima. 
# 7) Can also implement gradient descent if possible 

