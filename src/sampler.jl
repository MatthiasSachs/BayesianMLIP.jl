module Samplers


abstract type sampler

abstract type mhsampler <: sampler


function step(mhsampler, x)
    
end

end