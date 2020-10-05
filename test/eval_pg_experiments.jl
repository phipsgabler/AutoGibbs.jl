using Turing, Turing.RandomMeasures

# -- GMM example --
model = gmm_tarray_example()
disc_param = [:z]
cont_param = [:w, :μ]

# -- HMM example --
model = hmm_tarray_example()
disc_param = [:s]
cont_param = [:t, :m]

# -- trunctated IMM example --
data_neal = [-1.48, -1.40, -1.16, -1.08, -1.02, 0.14, 0.51, 0.53, 0.78]
α_neal = 10.0
Kmax = length(data_neal)
model = imm_stick_tarray_example(; K = Kmax)
disc_param = [:z]
cont_param = [:μ, :v]

# -- Parameters of inference algorithms --
particles = 10
n_step = 10
lf_size = 0.1

# -- Number of iterations --
iter = 5_000

# -- Inference --
alg = Gibbs(PG(particles, disc_param...), HMC(lf_size, n_step, cont_param...))
chn = sample(model, alg, iter);

# -- Show results --
@show alg
@show chn

