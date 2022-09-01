
using JuLIP, NeighbourLists
using JuLIP: bulk, rattle!, rnn
using ACEatoms: neighbourlist
using ACE
using ACEflux
# Define FS model
μ = ones()
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

rcut = rnn(:Cu) *1.5
linear_ace_layer = Linear_ACE(; ord = 2, maxdeg = 4, Nprop = 2, wL = 1.5,
                             Bsel = nothing, p = 1, rcut=rcut)
model = Chain(linear_ace_layer, GenLayer(FS), sum);
pot = ACEflux.FluxPotential(model, 6.0);

# extract ACE basis
basis = linear_ace_layer.m.basis

# generate atomic configuration
at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 


nlist = neighbourlist(at, r_cut)

# Get lists of atomic indices and discplacements corresponding to
# the atomic environemt of k-th atom: 
k = 1
Js, Rs = NeighbourLists.neigs(nlist, k) # Js = indices of neighbouring atoms, # Rs = displacement relative to the centre atom

# Convert list of discplacements to an ACE state
cfg = [ ACE.State(rr = r)  for (r) in Rs] |> ACEConfig
# evaluate ACE basis elements at k-th atomic environment:
B_vals = ACE.evaluate(basis, cfg)

# Get value of the k-th basis
k = 4
B_vals[k].val