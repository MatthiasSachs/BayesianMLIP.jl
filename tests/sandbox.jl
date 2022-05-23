using ACE
using ACEatoms
using BayesianMLIP.NLModels
using Random: seed!, rand
using JuLIP
#Import: Loads module. Names from module can be accessed with "Module.name" syntax. 
#Using: Loads module and makes the exported names available for direct use
#Export: Functions & structs that are exported are available for direct use when "using" a module 


maxdeg = 4 # max degree
ord = 2 # max body order correlation 
Bsel = SimpleSparseBasis(ord, maxdeg) #Structure with two attributes: maxlevel & maxorder 
rcut = 5.0 # Cutoff radius

# Create phi_mnl one particle basis
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, rin = 1.2, rcut = 5.0)
println(typeof(B1p))



