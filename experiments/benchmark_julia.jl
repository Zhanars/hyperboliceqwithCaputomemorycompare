#!/usr/bin/env julia
# Benchmark: FDM vs FEM vs Spectral on CGL nodes
# Time-fractional wave equation with Newmark-β=1/4 + L1 memory
# Usage: julia -t auto benchmark_julia.jl

using LinearAlgebra, SparseArrays, Printf
using SpecialFunctions: gamma
using Base.Threads

# ═══════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════
const L_domain = 1.0
const c_wave   = 1.0
const T_final  = 1.0
const γ_mem    = 1.0

# Exact solution: u(x,t) = t² sin(πx)
u_exact(x, t) = t^2 * sin(π * x)

function exact_rhs(x, t, α)
    s = sin(π * x)
    utt = 2.0 * s
    uxx = -π^2 * t^2 * s
    mem = 2γ_mem * t^(2 - α) / gamma(3 - α) * s
    return utt - c_wave^2 * uxx + mem
end

# ═══════════════════════════════════════════════════════════════
# Grid
# ═══════════════════════════════════════════════════════════════
function cgl_nodes(N)
    [L_domain / 2 * (1 - cos(π * j / N)) for j in 0:N]
end

function cc_weights(N)
    w = zeros(N + 1)
    for j in 0:N
        s = 0.0
        for k in 0:div(N, 2)
            bk = (k == 0 || k == N ÷ 2) ? 1.0 : 2.0
            ck = (k == 0 || k == N) ? 0.5 : 1.0
            s += bk * ck * cos(2π * k * j / N) / (1 - 4k^2)
        end
        cj = (j == 0 || j == N) ? 0.5 : 1.0
        w[j+1] = 2cj / N * s * (L_domain / 2)
    end
    return w
end

# ═══════════════════════════════════════════════════════════════
# FDM
# ═══════════════════════════════════════════════════════════════
function build_fdm(x)
    N = length(x) - 1
    n = N - 1
    dl = zeros(n - 1)
    dd = zeros(n)
    du = zeros(n - 1)
    for i in 1:n
        hm = x[i+1] - x[i]
        hp = x[i+2] - x[i+1]
        dd[i] = -2.0 / (hm * hp)
        if i > 1;  dl[i-1] = 2.0 / (hm * (hm + hp)); end
        if i < n;  du[i]   = 2.0 / (hp * (hm + hp)); end
    end
    return Tridiagonal(dl, dd, du)
end

# ═══════════════════════════════════════════════════════════════
# FEM
# ═══════════════════════════════════════════════════════════════
function build_fem(x)
    N = length(x) - 1
    n = N - 1
    h = diff(x)
    Md = [(h[i] + h[i+1]) / 3 for i in 1:n]
    Mo = [h[i+1] / 6 for i in 1:n-1]
    Kd = [1/h[i] + 1/h[i+1] for i in 1:n]
    Ko = [-1/h[i+1] for i in 1:n-1]
    M = Tridiagonal(copy(Mo), Md, copy(Mo))
    K = Tridiagonal(copy(Ko), Kd, copy(Ko))
    return M, K
end

# ═══════════════════════════════════════════════════════════════
# Spectral
# ═══════════════════════════════════════════════════════════════
function build_spectral(N)
    xi = [cos(π * j / N) for j in 0:N]
    cv = [((j == 0 || j == N) ? 2.0 : 1.0) * (-1.0)^j for j in 0:N]
    D = zeros(N+1, N+1)
    for i in 0:N, j in 0:N
        if i != j
            D[i+1, j+1] = cv[i+1] / (cv[j+1] * (xi[i+1] - xi[j+1]))
        end
    end
    for i in 0:N
        D[i+1, i+1] = -sum(D[i+1, :]) + D[i+1, i+1]
    end
    Dp = D * (-2.0 / L_domain)
    D2 = (Dp * Dp)[2:N, 2:N]
    return D2
end

# ═══════════════════════════════════════════════════════════════
# L1 Memory (standard, no SOE)
# ═══════════════════════════════════════════════════════════════
struct L1Memory
    α::Float64
    a0::Float64
    weights::Vector{Float64}
    history::Vector{Vector{Float64}}  # increments dU^k
end

function L1Memory(Nt, dt, ndof, α)
    a0 = dt^(-α) / gamma(2 - α)
    w = [(k+1)^(1-α) - k^(1-α) for k in 0:Nt]
    L1Memory(α, a0, w, Vector{Float64}[])
end

function push_eval!(mem::L1Memory, dU::Vector{Float64})
    push!(mem.history, copy(dU))
    n = length(mem.history)
    result = zeros(length(dU))
    for k in 1:n
        @inbounds result .+= mem.weights[n - k + 1] .* mem.history[k]
    end
    return mem.a0 * γ_mem .* result
end

# ═══════════════════════════════════════════════════════════════
# Solver: FDM
# ═══════════════════════════════════════════════════════════════
function solve_fdm(N, Nt, α)
    x = cgl_nodes(N)
    w = cc_weights(N)
    dt = T_final / Nt
    A = build_fdm(x)
    n_int = N - 1
    I_n = Matrix(1.0I, n_int, n_int)
    Lhs = I_n - (dt^2 * c_wave^2 / 4) * Matrix(A)
    Lhs_lu = lu(Lhs)
    mem = L1Memory(Nt, dt, n_int, α)

    U = zeros(Nt + 1, N + 1)
    U[2, 2:N] .= dt^2 .* sin.(π .* x[2:N])

    for n in 1:Nt-1
        dU = U[n+1, 2:N] .- U[n, 2:N]
        H = push_eval!(mem, dU)
        fn = [exact_rhs(x[i], n*dt, α) for i in 2:N]
        sA = (dt^2 * c_wave^2 / 4) * A * (2 .* U[n+1, 2:N] .+ U[n, 2:N])
        rhs = 2 .* U[n+1, 2:N] .- U[n, 2:N] .+ sA .+ dt^2 .* (fn .- H)
        U[n+2, 2:N] .= Lhs_lu \ rhs
    end

    ue = [u_exact(x[i], T_final) for i in 1:N+1]
    err_vec = U[Nt+1, :] .- ue
    l2err = sqrt(sum(w .* err_vec.^2))
    return l2err, U[Nt+1, :], x
end

# ═══════════════════════════════════════════════════════════════
# Solver: FEM
# ═══════════════════════════════════════════════════════════════
function solve_fem(N, Nt, α)
    x = cgl_nodes(N)
    w = cc_weights(N)
    dt = T_final / Nt
    M, K = build_fem(x)
    n_int = N - 1
    Lhs = Matrix(M) + (dt^2 * c_wave^2 / 4) * Matrix(K)
    Lhs_lu = lu(Lhs)
    mem = L1Memory(Nt, dt, n_int, α)

    U = zeros(Nt + 1, n_int)
    U[2, :] .= dt^2 .* sin.(π .* x[2:N])

    for n in 1:Nt-1
        dU = U[n+1, :] .- U[n, :]
        H = push_eval!(mem, dU)
        fn = [exact_rhs(x[i], n*dt, α) for i in 2:N]
        sK = -(dt^2 * c_wave^2 / 4) * K * (2 .* U[n+1, :] .+ U[n, :])
        rhs = M * (2 .* U[n+1, :] .- U[n, :]) .+ sK .+ dt^2 .* M * (fn .- H)
        U[n+2, :] .= Lhs_lu \ rhs
    end

    Uf = zeros(N + 1)
    Uf[2:N] .= U[Nt+1, :]
    ue = [u_exact(x[i], T_final) for i in 1:N+1]
    err_vec = Uf .- ue
    l2err = sqrt(sum(w .* err_vec.^2))
    return l2err, Uf, x
end

# ═══════════════════════════════════════════════════════════════
# Solver: Spectral
# ═══════════════════════════════════════════════════════════════
function solve_spectral(N, Nt, α)
    x = cgl_nodes(N)
    w = cc_weights(N)
    dt = T_final / Nt
    D2 = build_spectral(N)
    n_int = N - 1
    I_n = Matrix(1.0I, n_int, n_int)
    Lhs = I_n - (dt^2 * c_wave^2 / 4) * D2
    Lhs_lu = lu(Lhs)
    mem = L1Memory(Nt, dt, n_int, α)

    U = zeros(Nt + 1, n_int)
    U[2, :] .= dt^2 .* sin.(π .* x[2:N])

    for n in 1:Nt-1
        dU = U[n+1, :] .- U[n, :]
        H = push_eval!(mem, dU)
        fn = [exact_rhs(x[i], n*dt, α) for i in 2:N]
        sD = (dt^2 * c_wave^2 / 4) * D2 * (2 .* U[n+1, :] .+ U[n, :])
        rhs = 2 .* U[n+1, :] .- U[n, :] .+ sD .+ dt^2 .* (fn .- H)
        U[n+2, :] .= Lhs_lu \ rhs
    end

    Uf = zeros(N + 1)
    Uf[2:N] .= U[Nt+1, :]
    ue = [u_exact(x[i], T_final) for i in 1:N+1]
    err_vec = Uf .- ue
    l2err = sqrt(sum(w .* err_vec.^2))
    return l2err, Uf, x
end

# ═══════════════════════════════════════════════════════════════
# Parallel benchmark: run all three methods concurrently
# ═══════════════════════════════════════════════════════════════
function benchmark_parallel(N, Nt, α)
    results = Vector{Any}(undef, 3)
    times   = Vector{Float64}(undef, 3)

    @threads for i in 1:3
        t0 = time_ns()
        if i == 1
            results[i] = solve_fdm(N, Nt, α)
        elseif i == 2
            results[i] = solve_fem(N, Nt, α)
        else
            results[i] = solve_spectral(N, Nt, α)
        end
        times[i] = (time_ns() - t0) / 1e9
    end
    return results, times
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
function main()
    α = 0.5
    N = 16
    nthreads = Threads.nthreads()
    println("Julia $(VERSION), threads = $nthreads")
    println()

    # Warmup
    solve_fdm(4, 16, α); solve_fem(4, 16, α); solve_spectral(4, 16, α)
    benchmark_parallel(4, 16, α)

    # === Sequential benchmarks ===
    println("=== SEQUENTIAL BENCHMARK (N=$N, α=$α) ===")
    @printf("%6s | %8s | %10s | %10s | %10s | %10s | %10s | %10s\n",
            "Nt", "dt", "FDM err", "FDM time", "FEM err", "FEM time", "Spec err", "Spec time")
    println("-"^95)

    for Nt in [256, 512, 1024, 2048, 4096]
        t1 = @elapsed e1, _, _ = solve_fdm(N, Nt, α)
        t2 = @elapsed e2, _, _ = solve_fem(N, Nt, α)
        t3 = @elapsed e3, _, _ = solve_spectral(N, Nt, α)
        dt = T_final / Nt
        @printf("%6d | %8.5f | %10.3e | %8.4f s | %10.3e | %8.4f s | %10.3e | %8.4f s\n",
                Nt, dt, e1, t1, e2, t2, e3, t3)
    end

    # === Parallel benchmarks ===
    println()
    println("=== PARALLEL BENCHMARK (all 3 methods concurrent, N=$N, α=$α) ===")
    @printf("%6s | %10s | %10s | %10s | %10s\n",
            "Nt", "Wall time", "FDM time", "FEM time", "Spec time")
    println("-"^60)

    for Nt in [256, 512, 1024, 2048, 4096]
        wall_t = @elapsed res, times = benchmark_parallel(N, Nt, α)
        @printf("%6d | %8.4f s | %8.4f s | %8.4f s | %8.4f s\n",
                Nt, wall_t, times[1], times[2], times[3])
    end

    # === Speedup table ===
    println()
    println("=== SPEEDUP TABLE ===")
    @printf("%6s | %10s | %10s | %6s\n", "Nt", "Seq total", "Par wall", "Speedup")
    println("-"^42)

    for Nt in [256, 512, 1024, 2048, 4096]
        t1 = @elapsed solve_fdm(N, Nt, α)
        t2 = @elapsed solve_fem(N, Nt, α)
        t3 = @elapsed solve_spectral(N, Nt, α)
        seq_total = t1 + t2 + t3
        wall_t = @elapsed benchmark_parallel(N, Nt, α)
        speedup = seq_total / wall_t
        @printf("%6d | %8.4f s | %8.4f s | %5.2fx\n",
                Nt, seq_total, wall_t, speedup)
    end
end

main()
