module MHoutputschedulers 
using ACE 
using JuLIP
using BayesianMLIP.NLModels
using BayessianMLIP.Samplers
export MHoutp, feed!, greet
using ACE: val

struct MHoutp
    θ_steps
end
MHoutp() = MHoutp([]) 


function feed!(θ, outp::MHoutp)
    push!(outp.θ_steps, θ)
end 

end # end module