using LinearAlgebra: dot

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
