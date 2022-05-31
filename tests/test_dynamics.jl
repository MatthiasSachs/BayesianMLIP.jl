using JuLIP, Random
@info("Define (Morse) pair-potential")
r0 = rnn(:Al)
Vpair = JuLIP.morse(;A=4.0, e0=.5, r0=r0, rcut=(1.9*r0, 2.7*r0))

@info("Create random Al configuration")
Random.seed!(1234)
at = bulk(:Al, cubic=true) * 3
at = rattle!(at, 0.1)

# Compute forces 
forces(Vpair,at)
# Compute energy
energy(Vpair,at)