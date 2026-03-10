# =============================================================================
# Parallel Pairwise Distance Benchmark
# =============================================================================
#
# HOW TO REPRODUCE
# ─────────────────
# 1. Install Julia ≥ 1.10 from https://julialang.org/downloads/
#
# 2. Install the required package (one-time):
#      julia -e 'using Pkg; Pkg.add("BenchmarkTools")'
#
# 3. Run once for each desired thread count, e.g.:
#      julia -t 1  benchmark_parallel.jl
#      julia -t 2  benchmark_parallel.jl
#      julia -t 4  benchmark_parallel.jl
#      julia -t 8  benchmark_parallel.jl
#      julia -t 16 benchmark_parallel.jl
#
#    The script prints   Threads | Distances/sec   to stdout.
#    Collect those lines to feed into plot_scaling.jl.
#
# Note: GitHub Codespaces (free tier) provides only 2 virtual CPUs.
#       For a meaningful multi-core scaling curve, run on a local
#       workstation or a larger cloud instance.
# =============================================================================

using BenchmarkTools
using Base.Threads

include("parallel_euclid.jl")

const N = 10_000
points = rand(Float32, (N, 3))

println("Threads: $(Threads.nthreads())")
println("Points : $N")

# @belapsed returns the *minimum* elapsed time in seconds.
# Minimum is preferred here because it corresponds to the run with
# the least OS-scheduling interference — a standard HPC practice.
t = @belapsed pairwise_distances_parallel($points) samples=5 evals=1

# With the symmetry optimisation the number of *unique* distance
# calculations is N*(N+1)/2, not N^2.
ops = Int64(N) * (N + 1) ÷ 2

println("Min elapsed  : $(round(t, digits=4)) s")
println("Distances/sec: $(round(ops / t, sigdigits=4))")

# Machine-readable line for easy copy-paste into plot_scaling.jl
println("CSV: $(Threads.nthreads()),$(round(ops/t, sigdigits=6))")
