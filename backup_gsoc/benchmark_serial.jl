using BenchmarkTools
include("serial-euclid.jl")

const N = 10_000
points = rand(Float32, (N, 3))

println("=== Serial Benchmark ===")
println("Points: $N  |  Output matrix: $(N)×$(N) Float32  (~$(round(N*N*4/1e6)) MB)")
println()

# @benchmark (not @btime) gives full statistics.
# BenchmarkTools automatically warms up the function (runs it once to
# trigger JIT compilation) before recording any timings, so reported
# numbers reflect pure execution cost rather than compile time.
# The $ interpolation prevents Julia from treating `points` as a global
# variable, which would add spurious lookup overhead.
result = @benchmark pairwise_distances($points) samples=5 evals=1
display(result)

t_med = median(result).time / 1e9   # nanoseconds → seconds
ops   = Int64(N) * N                # total distance calculations
println()
println("Median time   : $(round(t_med, digits=3)) s")
println("Distances/sec : $(round(ops / t_med, sigdigits=4))")

println("""
─────────────────────────────────────────────
Notes on benchmarking methodology
─────────────────────────────────────────────
• JIT warm-up: Julia compiles each method on first call.
  BenchmarkTools executes a warm-up run before timing begins,
  so compilation cost is excluded from all reported figures.

• Interpolation (\$points): prevents global-variable lookup overhead
  that would inflate per-sample timings.

• Multiple samples (samples=5, evals=1): each sample is a full
  10k-point run (~250 ms); five samples give stable statistics
  without excessive wall-clock time.

• We report the *median* rather than the minimum because the
  median is more robust to occasional OS scheduling noise on a
  shared Codespaces VM.
─────────────────────────────────────────────
Inefficiencies in the serial version
─────────────────────────────────────────────
1. Redundant computation — the distance matrix is symmetric
   (d[i,j] == d[j,i]), yet every pair is computed twice.
   Restricting the inner loop to j ≥ i and mirroring halves
   the arithmetic and sqrt calls.

2. Single-core execution — the nested loop runs on one CPU
   core only, leaving all other cores idle.

3. Bounds-checking overhead — Julia inserts array-bounds checks
   on every subscript access.  @inbounds removes them inside the
   inner loop (safe here because i,j ∈ 1:n by construction).
─────────────────────────────────────────────
""")
