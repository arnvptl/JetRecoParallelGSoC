
"""
    pairwise_distances(points::AbstractArray)

Calculate the pairwise distances between 3D points in a an Nx3 array
"""
function pairwise_distances(points::AbstractArray{T}) where T
    @assert size(points)[2] == 3
    n = size(points)[1]
    distances = zeros(T, (n, n))
    for i in 1:n
        for j in 1:n
            dx = points[i, 1] - points[j, 1]
            dy = points[i, 2] - points[j, 2]
            dz = points[i, 3] - points[j, 3]
            distances[i, j] = sqrt(dx^2 + dy^2 + dz^2)
        end
    end
    return distances
end

function main()
    points_vector = rand(Float32, (10_000, 3))

    println("Starting calculation")
    distances = pairwise_distances(points_vector)
    println("Finished calculation")
end

main()

