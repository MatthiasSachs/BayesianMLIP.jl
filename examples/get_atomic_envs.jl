
using JuLIP, NeighbourLists, Flux, Plots
using JuLIP: bulk, rattle!, rnn
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Dynamics  
using ACEatoms: neighbourlist
using ACE
using ACEflux
# Define FS model
FS(ϕ) = ϕ[1] #+ sqrt(abs(ϕ[2]) + 1/100) - 1/10

rcut = rnn(:Cu) *1.5
linear_ace_layer = Linear_ACE(; ord = 2, maxdeg = 4, Nprop = 1, wL = 1.5,
                             Bsel = nothing, p = 1, rcut=rcut) ;
model = Chain(linear_ace_layer, GenLayer(FS), sum); 
pot = ACEflux.FluxPotential(model, rcut);
nparams(pot)
BayesianMLIP.NLModels.get_params(pot)

# extract ACE basis
basis = linear_ace_layer.m.basis

fieldnames(typeof(basis))

# generate atomic configuration
at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 
length(at)



BayesianMLIP.NLModels.set_params!(pot, ones(15))
energy(pot, at)
BayesianMLIP.NLModels.set_params!(pot, 2*ones(15))
energy(pot, at)

nlist = neighbourlist(at, rcut)

# Get lists of atomic indices and discplacements corresponding to
# the atomic environemt of i-th atom: 
i = 1
Js, Rs = NeighbourLists.neigs(nlist, i) # Js = indices of neighbouring atoms, # Rs = displacement relative to the centre atom
Js

# Convert list of discplacements to an ACE state
cfg = [ ACE.State(rr = r)  for (r) in Rs] |> ACEConfig
# evaluate ACE basis elements at i-th atomic environment:
B_vals = ACE.evaluate(basis, cfg)
array(B_vals)
sum(B_vals)

# Get value of the k-th basis 
k = 4
B_vals[k].val

k = 8
d = [] 
for i in 1:length(at) 
    Js, Rs = NeighbourLists.neigs(nlist, i)
    cfg = [ ACE.State(rr = r)  for (r) in Rs] |> ACEConfig
    B_vals = ACE.evaluate(basis, cfg)
    push!(d, B_vals[k].val)
end 
histogram(d, bins=:scott)

