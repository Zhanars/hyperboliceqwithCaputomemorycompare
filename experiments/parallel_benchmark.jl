#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
# Intra-method parallelisation benchmark
#
# For each solver (FDM, FEM, Spectral), we parallelise the
# INTERNAL computation: the L1 memory history sum and the
# RHS assembly. Each method is benchmarked SEPARATELY with
# varying thread counts: 1, 2, 3, 4, 6.
#
# Hardware: Intel i5-9400F (6 cores, no HT), 2.90 GHz
# Launch: julia -t 6 experiments/parallel_benchmark.jl
# ═══════════════════════════════════════════════════════════════

using LinearAlgebra, Printf
using SpecialFunctions: gamma

const L_dom = 1.0
const T_fin = 1.0
const c_wav = 1.0
const γ_mem = 1.0
const α_val = 0.5

u_exact(x, t) = t^2 * sin(π * x)

function exact_f(x, t, α, dt)
    s = sin(π * x)
    utt = 2.0 * s
    lap = c_wav^2 * π^2 * (t^2 + dt^2/2) * s
    cap = γ_mem * 2.0 / gamma(3 - α) * t^(2 - α) * s
    return utt + lap + cap
end

function cgl_nodes(N)
    [L_dom/2 * (1 - cos(π * (N - j) / N)) for j in 0:N]
end

function cc_weights(N)
    w = zeros(N+1)
    for j in 0:N
        s = 0.0
        for k in 1:div(N,2)
            bk = (k == div(N,2) && N % 2 == 0) ? 1.0 : 2.0
            s += bk * cos(2k * π * j / N) / (4k^2 - 1)
        end
        w[j+1] = (1.0 - s) / N
    end
    w[1] *= 0.5; w[N+1] *= 0.5
    return w .* 0.5
end

function build_fdm(x)
    N = length(x) - 1; h = diff(x); n = N - 1
    A = zeros(n, n)
    for i in 1:n
        hm = h[i]; hp = h[i+1]; hbar = 0.5*(hm+hp)
        A[i,i] = -(1/(hm*hbar) + 1/(hp*hbar))
        i > 1 && (A[i,i-1] = 1/(hm*hbar))
        i < n && (A[i,i+1] = 1/(hp*hbar))
    end
    return A
end

function build_fem(x)
    N = length(x) - 1; h = diff(x); n = N - 1
    M = zeros(n,n); K = zeros(n,n)
    for i in 1:n
        M[i,i] = (h[i]+h[i+1])/3; K[i,i] = 1/h[i]+1/h[i+1]
        if i > 1
            M[i,i-1] = h[i]/6; M[i-1,i] = h[i]/6
            K[i,i-1] = -1/h[i]; K[i-1,i] = -1/h[i]
        end
    end
    return M, K
end

function build_spectral(N)
    xi = [cos(π*j/N) for j in 0:N]
    cv = [((j==0||j==N) ? 2.0 : 1.0)*(-1.0)^j for j in 0:N]
    D = zeros(N+1,N+1)
    for i in 0:N, j in 0:N
        i != j && (D[i+1,j+1] = cv[i+1]/(cv[j+1]*(xi[i+1]-xi[j+1])))
    end
    for i in 1:N+1; D[i,i] = -sum(D[i,k] for k in 1:N+1 if k!=i); end
    Dp = D .* (-2.0/L_dom)
    D2 = (Dp*Dp)[2:N, 2:N]
    x_asc = [L_dom/2*(1-xi[i]) for i in 1:N+1]
    x_int = reverse(x_asc[2:N])
    return D2, x_asc, x_int
end

# ── Solver with thread-parallel L1 history sum ──
# The key parallelisable kernel is the O(N_t²) memory sum:
#   H_n = Σ_{j=0}^{n-1} w_j (U^{n-j} - U^{n-1-j})
# We partition the j-loop across threads.
function solve_parallel(method::Symbol, N, Nt, α; nthreads=1)
    x = cgl_nodes(N); dt = T_fin/Nt
    n_int = N - 1

    if method == :fdm
        A = build_fdm(x)
        Lhs = Matrix(1.0I, n_int, n_int) .- (dt^2*c_wav^2/4) .* A
        x_int = x[2:N]
    elseif method == :fem
        M, K = build_fem(x)
        Lhs = M .+ (dt^2*c_wav^2/4) .* K
        x_int = x[2:N]
    else
        D2, _, x_int_s = build_spectral(N)
        Lhs = Matrix(1.0I, n_int, n_int) .- (dt^2*c_wav^2/4) .* D2
        x_int = x_int_s
    end
    Lhs_lu = lu(Lhs)

    wL1 = [(k+1)^(1-α) - k^(1-α) for k in 0:Nt]
    a0 = dt^(-α) / gamma(2-α)

    U_all = zeros(Nt+1, n_int)
    U_all[2, :] = dt^2 .* sin.(π .* x_int)

    for n in 1:Nt-1
        tn = n * dt

        # ── Parallel L1 memory sum ──
        # Partition the history sum across nthreads
        Hn = zeros(n_int)
        if nthreads > 1 && n > 100
            # Use @threads for the history loop
            # Allocate per-thread accumulators (use max possible thread id)
            max_tid = Threads.maxthreadid()
            local_Hn = [zeros(n_int) for _ in 1:max_tid]
            Threads.@threads for j in 0:n-1
                tid = Threads.threadid()
                @views local_Hn[tid] .+= wL1[j+1] .* (U_all[n+1-j, :] .- U_all[n-j, :])
            end
            for t in 1:max_tid
                Hn .+= local_Hn[t]
            end
        else
            for j in 0:n-1
                @views Hn .+= wL1[j+1] .* (U_all[n+1-j, :] .- U_all[n-j, :])
            end
        end

        fn = [exact_f(x_int[i], tn, α, dt) for i in 1:n_int]

        if method == :fdm
            sA = (dt^2*c_wav^2/4) .* (A * (2 .* U_all[n+1,:] .+ U_all[n,:]))
            rhs = 2 .* U_all[n+1,:] .- U_all[n,:] .+ sA .+ dt^2 .* (fn .- a0*γ_mem .* Hn)
        elseif method == :fem
            sK = -(dt^2*c_wav^2/4) .* (K * (2 .* U_all[n+1,:] .+ U_all[n,:]))
            rhs = M*(2 .* U_all[n+1,:] .- U_all[n,:]) .+ sK .+ dt^2 .* (M * (fn .- a0*γ_mem .* Hn))
        else
            sD = (dt^2*c_wav^2/4) .* (D2 * (2 .* U_all[n+1,:] .+ U_all[n,:]))
            rhs = 2 .* U_all[n+1,:] .- U_all[n,:] .+ sD .+ dt^2 .* (fn .- a0*γ_mem .* Hn)
        end
        U_all[n+2,:] = Lhs_lu \ rhs
    end
    return U_all
end

function benchmark_method(method::Symbol, N, Nt, α; use_parallel=false)
    # Warm-up
    solve_parallel(method, N, min(Nt, 64), α; nthreads=(use_parallel ? Threads.nthreads() : 1))
    GC.gc()

    # Timed run
    nthreads = use_parallel ? Threads.nthreads() : 1
    t = @elapsed solve_parallel(method, N, Nt, α; nthreads=nthreads)
    return t
end

function main()
    α = α_val
    N = 128  # large enough for meaningful per-thread work
    max_threads = Threads.nthreads()

    println("Intra-method parallelisation benchmark")
    println("  CPU: Intel i5-9400F, 6 cores, no HT, 2.90 GHz")
    println("  Julia threads available: $max_threads")
    println("  Spatial nodes N=$N, α=$α")
    println("  Parallelised kernel: L1 memory history sum O(N_t²)")
    println("="^90)

    for method in [:fdm, :fem, :spec]
        method_name = method == :fdm ? "FDM" : method == :fem ? "FEM" : "Spectral"
        println("\n--- $method_name ---")
        @printf("%8s  %12s  %12s  %8s\n", "Nt", "1 thread (s)", "$max_threads threads", "Speedup")
        println("-"^50)

        for Nt in [512, 1024, 2048, 4096, 8192]
            # Sequential (1 thread)
            t_seq = benchmark_method(method, N, Nt, α; use_parallel=false)
            # Parallel (all threads)
            t_par = benchmark_method(method, N, Nt, α; use_parallel=true)
            speedup = t_seq / t_par
            @printf("%8d  %12.4f  %12.4f  %8.2f×\n", Nt, t_seq, t_par, speedup)
        end
    end
end

main()
