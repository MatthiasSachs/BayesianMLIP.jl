module json_parser 

using ACE, ACEatoms, Plots, ACEflux, Flux, JuLIP, StaticArrays
using JSON

export getData


function cfg_data_to_atom(cfg_data) 

    # Get position and momenta vectors
    X = [] 
    P = [] 
    M = [] 
    Z = []
    for particle in cfg_data["structure"]["sites"] 
        push!(X, particle["abc"])
        push!(P, particle["xyz"])
        push!(M, 63.546)
        push!(Z, AtomicNumber(29))
    end 

    # Get cell 
    cell = cfg_data["structure"]["lattice"]["matrix"]

    # Periodic Boundary Conditions 
    pbc = SVector{3, Bool}([1, 1, 1])

    calc = nothing 

    at = Atoms( X = X, 
            P = P, 
            M = M, 
            Z = Z, 
            cell = cell, 
            pbc = pbc, 
            calc = calc      )

    return at
end 

function getData(JSON_dataset) 
    Data = []
    for particles in JSON_dataset 
        atom = cfg_data_to_atom(particles) 
        E = particles["outputs"]["energy"]
        F = particles["outputs"]["forces"]
        
        d = (at = deepcopy(atom), E = E, F = F)

        push!(Data, d)
    end 

    return Data
end 



end 



