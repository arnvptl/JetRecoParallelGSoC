using Plots

# Thread counts tested
threads = [1, 2]

# Replace with your measured values
performance = [
    3.9200504610847634e8,
    4.395557367970841e8
]

# Generate plot
plot(
    threads,
    performance,
    xlabel="Thread Count",
    ylabel="Distance Measures per Second",
    title="Parallel Scaling of Pairwise Distance Computation",
    marker=:circle,
    linewidth=2,
    legend=false
)

# Save figure
savefig("scaling_plot.png")

println("Plot saved as scaling_plot.png")