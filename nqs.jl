using LinearAlgebra, Statistics
using TimerOutputs
using Distributions

using Random

const to = TimerOutput()

mutable struct NQSRBM{T1<:AbstractMatrix, T2<:AbstractVector}
    W::T1
    vbias::T2
    hbias::T2
end

function gen_state(config::AbstractVector, ansatz::NQSRBM)
    exp(transpose(ansatz.vbias) * config) * prod(cosh.(ansatz.hbias .+ transpose(ansatz.W) * config))
end

function gen_loc_energy(config::AbstractVector, ansatz::NQSRBM)
    type = eltype(ansatz.vbias)
    field = one(type) # external field
    flipterm = zero(type)
    for i in eachindex(config)
        config[i] *= -1; flipterm += gen_state(config, ansatz); config[i] *= -1 # flip back
    end
    return -field * flipterm / gen_state(config, ansatz) - dot(config[1:end-1], config[2:end])
end


function gen_derivs(config::AbstractVector, ansatz::NQSRBM)
    vsize = length(ansatz.vbias)
    hsize = length(ansatz.hbias)
    dW = vec(config * transpose(tanh.(ansatz.hbias + vec(config' * ansatz.W))))
    dvbias = typeof(ansatz.vbias)(config)
    dhbias = tanh.(ansatz.hbias + vec(config' * ansatz.W))
    return vcat(dW, dvbias, dhbias)
end


# single spin flip
function metropolis(s0, ansatz)
    idx = rand(1:length(ansatz.vbias))
    s1 = copy(s0); s1[idx] *= -1
    ratio = abs(gen_state(s1, ansatz)/gen_state(s0, ansatz))^2
    if ratio > 1.0
        s1
    else
        rand() < ratio ? s1 : s0
    end
end


function sampler(s0, ansatz, n_samples=1000; n_decor_steps=length(ansatz.vbias))
    vsize = length(ansatz.vbias)

    configs = Array{typeof(s0)}(undef, n_samples)

    for i in 1:n_samples
        for j in 1:n_decor_steps # default to a sweep
            s0 = metropolis(s0, ansatz)
        end
        configs[i] = s0
    end
    return configs
end

function burnout(s0, ansatz; n_steps=1000) 
    for i in 1:n_steps
        s0 = metropolis(s0, ansatz)
    end
    return s0
end


# utility functions to check samples
function samples_to_probs(samples)
    to_num(d) = (d = Array{Bool}((d .+ 1)/2); sum([d[k]*2^(k-1) for k=1:length(d)])) # digits to num
    sprobs = zeros(Int, 2^length(samples[1]))
    for s in samples
       global sprobs[to_num(s)+1] += 1
    end
    return sprobs ./ sum(sprobs)
end

myseed = 31415926
Random.seed!(myseed)

println(stdout, "# seed $(myseed)")

type = Complex{Float64}
#type = Float64

#vsize = 32
#hsize = 64
vsize = 16
hsize = 16

#n_samples = 1000
#n_iters = 4000
n_samples = 1000
n_iters = 1000

scale = 10.0

lamb_0 = 100.0
lamb_min = 1e-4
lamb_b = 0.9

println(stdout, "# type $(type)")
println(stdout, "# vsize $(vsize)")
println(stdout, "# hsize $(hsize)")
println(stdout, "# n_samples $(n_samples)")
println(stdout, "# n_iters $(n_iters)")

ansatz = NQSRBM(1e-1randn(type, vsize, hsize), 1e-1randn(type, vsize), 1e-1randn(type, hsize))



s0 = ones(Int, vsize)
s0 = burnout(s0, ansatz, n_steps=1000)


for iter in 1:n_iters

@timeit to "sampler" begin
s0 = burnout(s0, ansatz, n_steps=div(n_samples, 10)) # burnout each time by 10%
configs = sampler(s0, ansatz, n_samples)
global s0 = configs[end] # persistent chain
end


@timeit to "states" begin
states = gen_state.(configs, Ref(ansatz))
states = conj.(states) .* states
probs = states ./ sum(states)
end


@timeit to "derivs" begin
# of size parameters size x configs size
derivs_configs = hcat(gen_derivs.(configs, Ref(ansatz))...)
derivs = derivs_configs * probs
end

@timeit to "covariance" begin
# covariance with respect to probs (wave function probs)
derivs_configs_centered = derivs_configs .- derivs_configs * probs
sqrtprobs = derivs_configs_centered .* sqrt.(probs)'
covariance = conj.(sqrtprobs) * transpose(sqrtprobs)

# regularization
covariance += max(lamb_0 * lamb_b^iter, lamb_min) * Diagonal(covariance)
end


@timeit to "energy" begin
loc_energy_configs = gen_loc_energy.(configs, Ref(ansatz))
energy = dot(probs, loc_energy_configs)
end

println(stdout, "$iter, $(real(energy/vsize))")
flush(stdout)

@timeit to "force" begin
forces = sum([probs[i] * loc_energy_configs[i] * conj.(derivs_configs[:, i]) for i in 1:n_samples])
forces -= energy * conj.(derivs)
end


@timeit to "inverse" begin
updates = pinv(covariance) * forces
updates /= scale - energy - transpose(derivs) * updates
end


@timeit to "updates" begin
ansatz.W -= reshape(updates[1:vsize*hsize], (vsize, hsize))
ansatz.vbias -= updates[vsize*hsize+1: vsize*hsize+vsize]
ansatz.hbias -= updates[vsize*hsize+vsize+1: vsize*hsize+vsize+hsize]
end

end


print_timer(stderr, to)
println()
