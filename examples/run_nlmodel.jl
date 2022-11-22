using ACE, ACEatoms, Flux, ACEflux, LinearAlgebra, JLD2, JuLIP, StaticArrays, Statistics, JSON, Plots, BenchmarkTools
using BayesianMLIP, BayesianMLIP.NLModels, BayesianMLIP.Utils, BayesianMLIP.Outputschedulers, BayesianMLIP.json_parser, BayesianMLIP.globalSamplers, BayesianMLIP.conditionalSamplers
import ACEflux: FluxPotential 
import Distributions: MvNormal
using ACE: O3, SymmetricBasis, LinearACEModel, params

real_data = getData(JSON.parsefile("./Run_Data/Real_Data/training_test/Cu/training.json"))[1:end] ; 
at = real_data[1].at ;

# Construct ACE basis w/ appropriate hyperparameters
order = 1; maxdeg = 5; 
r0 = rnn(:Cu) ; rcut = 10.0
Bsel = SimpleSparseBasis(order, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(;
    maxdeg = maxdeg, 
    Bsel = Bsel, 
    r0 = r0, 
    trans = PolyTransform(2, r0),
    rin = 0.65*r0, 
    rcut = rcut, 
    pcut = 2,
    pin = 2, 
    constants = false
);
φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

# Construct LinearACEModel with 2 copies of paramters 
Nprop = 2;
c = zeros(Nprop, length(basis));
lin_model = LinearACEModel(basis, ACEflux.matrix2svector(c); evaluator = :standard)

# Construct nonlinear model 
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10; 
nl_model = Chain(Linear_ACE(c, lin_model), GenLayer(FS)); 
pot = FluxPotential(nl_model, rcut); 

nparams(pot)
nlinparams(pot)
get_params(pot)

twoK = nparams(pot)

Mu = zeros(twoK); sig = randn(twoK, twoK); Sig = sig'*sig; 
log_likelihood_L2(pot::ACEflux.FluxPotential, d; ωE = 1.0, ωF = ωE/(3*length(d.at)) ) = -ωE * (d.E - energy(pot, d.at))^2 -  ωF * sum(sum(abs2, g - f) for (g, f) in zip(forces(pot, d.at), d.F))

log_likelihood_Null = ConstantLikelihood() 
priorNormal = MvNormal(Mu, Sig)
priorUniform = FlatPrior()
real_data = getData(JSON.parsefile("./Run_Data/Real_Data/training_test/Cu/training.json"))[1:end] ; 

stm = StatisticalModel(log_likelihood_L2, priorUniform, pot, real_data) ;
begin 
    # Check that setting all lin params = 1, nonlin=0 and evaluating energy gives equal energy value as sum of 1st row of design matrix 
    set_params!(pot, vcat(ones(6), zeros(6)))
    @show energy(pot, at) == sum(design_matrix(stm)[1, :])
end 

st = State_θ(randn(twoK), randn(twoK)) 
output = outp() ; 

s = linearMetropolis(1e-1, st, stm) ;
s = linearSGLD(1e-1, st, stm, 1; β=1.0, α=0.9, transf_mean = zeros(twoK), transf_std = Diagonal(ones(twoK))) ;
conditionalSamplers.run!(st, s, stm, 1000, output) 
Summary(output)
Histogram(output, [1, 2, 3, 4, 5, 6])
Trajectory(output) 

scatter(
    [x[1] for x in output.θ], 
    [x[2] for x in output.θ], 
    legend=false, 
    markersize=1, 
)


st1 = State_θ(vcat([-91.23915391, -66.78643017, 66.45443175], randn(3))) 
output1 = outp() ; 
s1 = nonlinearSGLD(1e-14, st1, stm, 1; β=1e-9, α=1.0) ;
conditionalSamplers.run!(st1, s1, stm, 10000, output1) 
delete_first!(output1, 1000)

Summary(output1)
Histogram(output1)
Trajectory(output1) 




D = svd(Sig); STD = D.U * Diagonal(sqrt.(D.S)) * transpose(D.U)

x = ones(6)
lp1 = get_lp(stm, Mu, STD);
lp2 = get_lp(stm, zeros(6), Diagonal(ones(6)));
glp1 = get_glp(stm, Mu, STD) ;
glp2 = get_glp(stm, zeros(6), Diagonal(ones(6))) ;

lp1(x, stm.data, length(stm.data))==lp2(STD * x + Mu, stm.data, length(stm.data))

glp1(x, stm.data, length(stm.data))
STD * (glp2(STD * x + Mu, stm.data, length(stm.data)))

