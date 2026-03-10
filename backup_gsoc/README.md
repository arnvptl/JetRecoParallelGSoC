

# Parallel Processing Improvements in Julia Jet Reconstruction

## GSoC 2026 Evaluation Exercise

This repository contains the evaluation exercise for candidates interested in the HSF/CERN GSoC project [*Parallel Processing Improvements in Julia Jet Reconstruction*](https://hepsoftwarefoundation.org/gsoc/2026/proposal_JuliaHEP_JetReconstruction.html).

## Instructions

1. Please get in touch with the [mentors of the project](https://hepsoftwarefoundation.org/gsoc/2026/proposal_JuliaHEP_JetReconstruction.html) to register your interest.
2. Read the task instructions below carefully.
3. Fork the repository and work on your solution.
    - You may set your fork to *private*, if you wish.
4. Invite the mentors to look at your solution *by 16 March*.
    - We will give you some feedback and advice on whether we recommend you to proceed with a proposal for project.

## Task

In this repository you will find a Julia script, `serial-euclid.jl` that
calculates pairwise Euclidean distances between a large number of points.

- Make sure you can setup Julia and run the code.

### Benchmark Serial Version

- Your first task is to benchmark the initial serial version of the code, using
standard Julia tools.

- Comment on
    - how the benchmarking is done and why, with reference to
    warm-ups, JIT and any other relevant factors;
    - the efficiency of this serial version and on any obvious ways to
    improve it.

### Develop a Parallelisation Strategy

- Now you should implement a parallel version of the code in Julia that can run
on multiple CPU cores.

- Make sure you benchmark the performance, as a function of the number of
threads.

- Please produce a plot of distance-measures-per-second vs. thread count and
comment on the results you find.

Your parallel version of the benchmarking code should contain simple
instructions for how to reproduce the results (we will fork it and follow your
instructions as part of the evaluation).

### Discussion

- Now imagine you now have to port this code to a GPU, using Julia. What would
be the key things to pay attention to to ensure the performance is optimal?

## Regarding AI

It is permitted to use AI to help you in this project, but please *do not use
coding assistants to generate your solution*. Please include a statement saying
to what extent you used AI tools when undertaking the exercise.
