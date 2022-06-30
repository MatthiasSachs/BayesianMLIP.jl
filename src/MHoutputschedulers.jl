module MHoutputschedulers 
using ACE 
using ACE: val
using JuLIP
export feed!

abstract type mhoutputscheduler end

struct MHoutp <: mhoutputscheduler
    θ_steps
    log_posterior::Vector{Float64}
end
MHoutp() = MHoutp([], []) 


function feed!(θ, outp::MHoutp)
    push!(outp.θ_steps, θ)
end 

end # end module