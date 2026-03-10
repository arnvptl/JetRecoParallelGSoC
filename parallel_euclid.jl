using Base.Threads

"""
    pairwise_distances_parallel(points::AbstractArray)

Parallel computation of pairwise Euclidean distances between 3D points
stored in an N×3 array.

Optimisations over the serial version:
  1. Exploits the symmetry d(i,j) == d(j,i): only the upper triangle
     (j ≥ i) is computed; the result is mirrored immediately, halving
     the number of sqrt calls and arithmetic operations.
  2. The outer loop is parallelised with Threads.@threads.
     Thread-safety: thread i owns row-i writes (distances[i,j]) and
     column-i writes (distances[j,i]).  Because different values of i
     never produce the same (row,col) pair there are no data races.
  3. @inbounds suppresses redundant bounds checks inside the hot loop.
"""
function pairwise_distances_parallel(points::AbstractArray{T}) where T
    @assert size(points, 2) == 3
    n = size(points, 1)
    distances = zeros(T, n, n)

    @threads for i in 1:n
        @inbounds for j in i:n
            dx = points[i, 1] - points[j, 1]
            dy = points[i, 2] - points[j, 2]
            dz = points[i, 3] - points[j, 3]
            d  = sqrt(dx*dx + dy*dy + dz*dz)
            distances[i, j] = d
            distances[j, i] = d
        end
    end

    return distances
end
