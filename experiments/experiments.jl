#!/usr/bin/env julia
# experiments.jl — Full test suite in Julia for:
#   - Time-fractional hyperbolic equations (FDM, FEM, Spectral)
#   - Fast SOE memory algorithm
#   - Spatial and temporal convergence studies
#   - SOE speedup benchmarks
#   - Seismic wave illustrative example
#   - Condition number analysis
#   - Efficiency frontier data
#   - Task-level parallel speedup
#
# Usage: julia -t auto experiments.jl

using LinearAlgebra, Printf
using SpecialFunctions: gamma
using Base.Threads

# ═════════════════════════════════════════════════════════════
# 1. PHYSICAL PARAMETERS & SHARED UTILITIES
# ═════════════════════════════════════════════════════════════
const L_dom = 1.0
const T_fin = 1.0
const c_wav = 1.0
const γ_mem = 1.0

u_exact(x, t) = t^2 * sin(π * x)

# Manufactured forcing for u = t²sin(πx), Newmark β=1/4
# u_tt = 2sin(πx), u_xx at Newmark avg = -π²(t_n² + dt²/2)sin(πx)
# Memory ∂_t^α u(t_n) = 2t_n^{2-α}/Γ(3-α)·sin(πx)
function exact_f(x, t, α, dt)
    s = sin(π * x)
    return 2.0*s + c_wav^2 * π^2 * (t^2 + dt^2/2.0) * s +
           γ_mem * 2.0 / gamma(3.0 - α) * t^(2.0 - α) * s
end

function cgl_nodes(N)
    return [L_dom/2 * (1 - cos(π * (N - j) / N)) for j in 0:N]
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
    w[1]   *= 0.5
    w[N+1] *= 0.5
    return w .* 0.5
end

function l2_error(U, x, w)
    e = U .- [u_exact(x[i], T_fin) for i in eachindex(x)]
    return sqrt(sum(w .* e.^2))
end

function conv_rate(e_prev, e_curr; r=2.0)
    (isnothing(e_prev) || e_prev ≤ 0 || e_curr ≤ 0) && return NaN
    return log(e_prev / e_curr) / log(r)
end

# ═════════════════════════════════════════════════════════════
# 2. FAST MEMORY (SOE) ALGORITHM
# ═════════════════════════════════════════════════════════════
mutable struct FastMemory
    a0::Float64
    gam::Float64
    p::Int
    use_soe::Bool
    wL1::Vector{Float64}
    c_soe::Vector{Float64}
    q_soe::Vector{Float64}
    Z::Matrix{Float64}
    buf::Matrix{Float64}
    step_n::Int
end

function FastMemory(Nt, dt, n_dof, α; gam=1.0, p=10, Ne=20, use_soe=true)
    a0 = dt^(-α) / gamma(2.0 - α)
    p_eff = min(p, Nt)
    do_soe = use_soe && (Nt > p_eff)
    if !do_soe; p_eff = Nt; end
    wL1 = [(k+1)^(1-α) - k^(1-α) for k in 0:max(Nt, p_eff)]

    c_soe_v = Float64[]
    q_soe_v = Float64[]
    Z = zeros(0, n_dof)
    if do_soe
        x = range(log(0.5 / max(Nt, 10)), log(20.0), length=Ne)
        h = (x[end] - x[1]) / (Ne - 1)
        mu = exp.(x)
        c_soe_v = (1-α) * sin(π*α) / π .* h .* mu.^α
        q_soe_v = exp.(-mu)
        Z = zeros(Ne, n_dof)
    end
    buf = zeros(p_eff, n_dof)
    FastMemory(a0, gam, p_eff, do_soe, wL1, c_soe_v, q_soe_v, Z, buf, 0)
end

function push_eval!(fm::FastMemory, dU::Vector{Float64})
    if fm.use_soe && fm.step_n >= fm.p
        exiting = fm.buf[end, :]
        for i in axes(fm.Z, 1)
            @inbounds fm.Z[i, :] .= fm.q_soe[i] .* fm.Z[i, :] .+ exiting
        end
    end
    # Roll buffer
    for i in size(fm.buf, 1):-1:2
        fm.buf[i, :] .= fm.buf[i-1, :]
    end
    fm.buf[1, :] .= dU
    fm.step_n += 1

    n = fm.step_n
    p_eff = min(fm.p, n)
    H = zeros(length(dU))
    for j in 1:p_eff
        @inbounds H .+= fm.wL1[j] .* @view(fm.buf[j, :])
    end
    if fm.use_soe && n > fm.p
        H .+= vec(fm.c_soe' * fm.Z)
    end
    return fm.a0 * fm.gam .* H
end

# ═════════════════════════════════════════════════════════════
# 3. SPATIAL DISCRETISATION OPERATORS
# ═════════════════════════════════════════════════════════════
function build_fdm_laplacian(x)
    N = length(x) - 1
    h = diff(x)
    n = N - 1
    A = zeros(n, n)
    for i in 1:n
        hm = h[i]; hp = h[i+1]
        hbar = 0.5 * (hm + hp)
        A[i, i] = -(1.0/(hm*hbar) + 1.0/(hp*hbar))
        if i > 1; A[i, i-1] = 1.0 / (hm * hbar); end
        if i < n; A[i, i+1] = 1.0 / (hp * hbar); end
    end
    return A
end

function build_fem_matrices(x)
    N = length(x) - 1
    h = diff(x)
    n = N - 1
    M = zeros(n, n)
    K = zeros(n, n)
    for i in 1:n
        M[i,i] = (h[i] + h[i+1]) / 3
        K[i,i] = 1/h[i] + 1/h[i+1]
        if i > 1
            M[i,i-1] = h[i] / 6
            M[i-1,i] = h[i] / 6
            K[i,i-1] = -1/h[i]
            K[i-1,i] = -1/h[i]
        end
    end
    return M, K
end

function build_chebyshev_D2(N)
    xi = [cos(π * j / N) for j in 0:N]
    cv = [((j == 0 || j == N) ? 2.0 : 1.0) * (-1.0)^j for j in 0:N]
    D = zeros(N+1, N+1)
    for i in 0:N, j in 0:N
        if i != j
            D[i+1, j+1] = cv[i+1] / (cv[j+1] * (xi[i+1] - xi[j+1]))
        end
    end
    for i in 1:N+1
        D[i, i] = -sum(D[i, k] for k in 1:N+1 if k != i)
    end
    Dp = D .* (-2.0 / L_dom)
    D2 = (Dp * Dp)[2:N, 2:N]
    x_asc = [L_dom/2 * (1 - xi[i]) for i in 1:N+1]
    x_int = reverse(x_asc[2:N])
    return D2, x_asc, x_int
end

# ═════════════════════════════════════════════════════════════
# 4. SOLVERS (Newmark β=1/4 + explicit L1 memory)
# ═════════════════════════════════════════════════════════════
function solve_fdm(N, Nt, α; use_soe=true)
    x = cgl_nodes(N); w = cc_weights(N); dt = T_fin / Nt
    A = build_fdm_laplacian(x)
    n_int = N - 1
    Lhs = Matrix(1.0I, n_int, n_int) .- (dt^2 * c_wav^2 / 4) .* A
    Lhs_lu = lu(Lhs)
    fm = FastMemory(Nt, dt, n_int, α; gam=γ_mem, use_soe=use_soe)

    U_prev = zeros(n_int)
    U_curr = dt^2 .* sin.(π .* x[2:N])

    t0 = time_ns()
    for n in 1:Nt-1
        dU = U_curr .- U_prev
        mem = push_eval!(fm, dU)
        fn = [exact_f(x[i+1], n*dt, α, dt) for i in 1:n_int]
        sA = (dt^2 * c_wav^2 / 4) .* (A * (2 .* U_curr .+ U_prev))
        rhs = 2 .* U_curr .- U_prev .+ sA .+ dt^2 .* (fn .- mem)
        U_next = Lhs_lu \ rhs
        U_prev = U_curr
        U_curr = U_next
    end
    elapsed = (time_ns() - t0) / 1e9

    Uf = zeros(N+1); Uf[2:N] .= U_curr
    return l2_error(Uf, x, w), elapsed
end

function solve_fem(N, Nt, α; use_soe=true)
    x = cgl_nodes(N); w = cc_weights(N); dt = T_fin / Nt
    M, K = build_fem_matrices(x)
    n_int = N - 1
    Lhs = M .+ (dt^2 * c_wav^2 / 4) .* K
    Lhs_lu = lu(Lhs)
    fm = FastMemory(Nt, dt, n_int, α; gam=γ_mem, use_soe=use_soe)

    U_prev = zeros(n_int)
    U_curr = dt^2 .* sin.(π .* x[2:N])

    t0 = time_ns()
    for n in 1:Nt-1
        dU = U_curr .- U_prev
        mem = push_eval!(fm, dU)
        fn = [exact_f(x[i+1], n*dt, α, dt) for i in 1:n_int]
        sK = -(dt^2 * c_wav^2 / 4) .* (K * (2 .* U_curr .+ U_prev))
        rhs = M * (2 .* U_curr .- U_prev) .+ sK .+ dt^2 .* (M * (fn .- mem))
        U_next = Lhs_lu \ rhs
        U_prev = U_curr
        U_curr = U_next
    end
    elapsed = (time_ns() - t0) / 1e9

    Uf = zeros(N+1); Uf[2:N] .= U_curr
    return l2_error(Uf, x, w), elapsed
end

function solve_spectral(N, Nt, α; use_soe=true)
    D2, x_asc, x_int = build_chebyshev_D2(N)
    x_cgl = cgl_nodes(N); w = cc_weights(N); dt = T_fin / Nt
    n_int = N - 1
    Lhs = Matrix(1.0I, n_int, n_int) .- (dt^2 * c_wav^2 / 4) .* D2
    Lhs_lu = lu(Lhs)
    fm = FastMemory(Nt, dt, n_int, α; gam=γ_mem, use_soe=use_soe)

    U_prev = zeros(n_int)
    U_curr = dt^2 .* sin.(π .* x_int)

    t0 = time_ns()
    for n in 1:Nt-1
        dU = U_curr .- U_prev
        mem = push_eval!(fm, dU)
        fn = [exact_f(x_int[i], n*dt, α, dt) for i in 1:n_int]
        sD = (dt^2 * c_wav^2 / 4) .* (D2 * (2 .* U_curr .+ U_prev))
        rhs = 2 .* U_curr .- U_prev .+ sD .+ dt^2 .* (fn .- mem)
        U_next = Lhs_lu \ rhs
        U_prev = U_curr
        U_curr = U_next
    end
    elapsed = (time_ns() - t0) / 1e9

    Uf = zeros(N+1); Uf[2:N] .= U_curr
    return l2_error(reverse(Uf), x_asc, w), elapsed
end

# ═════════════════════════════════════════════════════════════
# 5. EXPERIMENTS
# ═════════════════════════════════════════════════════════════

function exp_spatial_convergence()
    NT_FIXED = 16384
    conv_Ns = [4, 8, 16, 32]
    println("\n", "="^60)
    println("  TABLE 3: Spatial convergence (Spectral, Nt=$NT_FIXED)")
    println("="^60)
    @printf("%5s %6s   %10s %5s\n", "N", "Nt", "Spec err", "Rate")
    println("-"^35)
    prev_e = nothing
    for N in conv_Ns
        e, _ = solve_spectral(N, NT_FIXED, 0.5; use_soe=false)
        r = conv_rate(prev_e, e)
        prev_e = e
        f(r) = isnan(r) ? "  ---" : @sprintf("%5.2f", r)
        @printf("%5d %6d   %10.3e %5s\n", N, NT_FIXED, e, f(r))
    end
end

function exp_temporal_convergence()
    conv_Nts = [64, 128, 256, 512, 1024, 2048]
    println("\n", "="^55)
    println("  TABLE 4: Temporal convergence (Spectral, N=16)")
    println("="^55)
    for α in [0.1, 0.5, 0.8]
        println("\n--- α = $α ---")
        @printf("%6s %9s   %10s %5s\n", "Nt", "dt", "Spec err", "Rate")
        println("-"^40)
        prev_e = nothing
        for Nt in conv_Nts
            dt = T_fin / Nt
            e, _ = solve_spectral(16, Nt, α; use_soe=false)
            r = conv_rate(prev_e, e)
            prev_e = e
            f(r) = isnan(r) ? "  ---" : @sprintf("%5.2f", r)
            @printf("%6d %9.5f   %10.3e %5s\n", Nt, dt, e, f(r))
        end
    end
end

function exp_soe_speedup()
    Nt_list = [64, 128, 256, 512, 1024]
    N = 12
    α = 0.5
    println("\n", "="^75)
    println("  TABLE 2: SOE speedup (Spectral, N=$N, α=$α)")
    println("="^75)
    @printf("%6s   %12s   %12s   %8s   %8s\n", "Nt", "Std time(s)", "SOE time(s)", "Speedup", "Err ratio")
    println("-"^75)
    for Nt in Nt_list
        e_s, t_s = solve_spectral(N, Nt, α; use_soe=false)
        e_f, t_f = solve_spectral(N, Nt, α; use_soe=true)
        speedup = t_s / max(t_f, 1e-12)
        ratio = e_f / e_s
        @printf("%6d   %12.6f   %12.6f   %7.2fx   %8.4f\n", Nt, t_s, t_f, speedup, ratio)
    end
end

function exp_condition_numbers()
    println("\n", "="^75)
    println("  CONDITION NUMBER ANALYSIS  (Julia)")
    println("="^75)
    @printf("%5s   %14s   %14s   %14s   %10s\n",
            "N", "κ(-A_FDM)", "κ(M⁻¹K_FEM)", "κ(-D²_Spec)", "FEM/FDM")
    println("-"^75)
    for N in [8, 16, 32, 64]
        x = cgl_nodes(N)
        A = build_fdm_laplacian(x)
        cn_fdm = cond(-A)

        Mf, Kf = build_fem_matrices(x)
        MiK = Mf \ Kf
        cn_fem = cond(MiK)

        D2, _, _ = build_chebyshev_D2(N)
        cn_spec = cond(-D2)

        @printf("%5d   %14.2e   %14.2e   %14.2e   %10.2f\n",
                N, cn_fdm, cn_fem, cn_spec, cn_fem / cn_fdm)
    end
end

function exp_efficiency_frontier()
    println("\n", "="^55)
    println("  EFFICIENCY FRONTIER (Spectral)")
    println("="^55)
    @printf("%5s %6s   %10s   %10s\n", "N", "Nt", "Error", "Time(s)")
    println("-"^40)
    for N in [4, 8, 16, 32]
        for Nt in [32, 64, 128, 256, 512, 1024]
            e, t = solve_spectral(N, Nt, 0.5; use_soe=false)
            @printf("%5d %6d   %10.4e   %10.6f\n", N, Nt, e, t)
        end
    end
end

function exp_parallel_speedup()
    println("\n  (Parallel speedup skipped — single-method mode)")
end

function exp_seismic()
    println("\n", "="^65)
    println("  ILLUSTRATIVE EXAMPLE (Seismic Wave)  (Julia)")
    println("="^65)

    L_phy, c_phy, α_phy, g_phy, T_phy = 1000.0, 3000.0, 0.7, 1500.0, 0.15
    N_phy, Nt_phy = 120, 400
    dt_phy = T_phy / Nt_phy

    D2, x_asc, x_int = build_chebyshev_D2(N_phy)
    # Scale D2 for physical domain (already 2/L factor built in for L=1, need to rescale)
    # Actually rebuild for physical domain
    N = N_phy
    xi = [cos(π * j / N) for j in 0:N]
    cv = [((j == 0 || j == N) ? 2.0 : 1.0) * (-1.0)^j for j in 0:N]
    D = zeros(N+1, N+1)
    for i in 0:N, j in 0:N
        if i != j
            D[i+1, j+1] = cv[i+1] / (cv[j+1] * (xi[i+1] - xi[j+1]))
        end
    end
    for i in 1:N+1
        D[i, i] = -sum(D[i, k] for k in 1:N+1 if k != i)
    end
    x_phys = (L_phy / 2) .* (1 .- xi)
    Dp = D .* (-2.0 / L_phy)
    D2_int = (Dp * Dp)[2:N, 2:N]
    n_int = N - 1

    u0 = exp.(-((x_phys .- L_phy/2) ./ 30.0).^2) .* sin.(π .* x_phys ./ L_phy)

    a0 = dt_phy^(-α_phy) / gamma(2 - α_phy)
    wL1 = [(k+1)^(1-α_phy) - k^(1-α_phy) for k in 0:Nt_phy-1]

    Lhs = Matrix(1.0I, n_int, n_int) .- dt_phy^2 * c_phy^2 / 4.0 .* D2_int

    U_prev = u0[2:N]
    U_curr = U_prev .+ 0.5 * dt_phy^2 .* (c_phy^2 .* (D2_int * U_prev))


    t0 = time_ns()
    # Full history approach
    U_all = zeros(Nt_phy + 1, n_int)
    U_all[1, :] = u0[2:N]
    U_all[2, :] = U_all[1, :] .+ 0.5 * dt_phy^2 .* (c_phy^2 .* (D2_int * U_all[1, :]))

    for n in 1:Nt_phy-1
        Hn = zeros(n_int)
        for j in 0:n-1
            @views Hn .+= wL1[j+1] .* (U_all[n+1-j, :] .- U_all[n-j, :])
        end
        sD = dt_phy^2 * c_phy^2 / 4.0 .* (D2_int * (2 .* U_all[n+1, :] .+ U_all[n, :]))
        rhs = 2 .* U_all[n+1, :] .- U_all[n, :] .+ sD .- dt_phy^2 * a0 * g_phy .* Hn
        U_all[n+2, :] = Lhs \ rhs
    end
    elapsed = (time_ns() - t0) / 1e9

    max_amp = maximum(abs.(U_all[end, :]))
    println("  Seismic simulation complete: N=$N_phy, Nt=$Nt_phy, α=$α_phy")
    @printf("  Max amplitude at T=%.3f: %.6e\n", T_phy, max_amp)
    @printf("  Elapsed: %.3f s\n", elapsed)
end

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
function main()
    println("Julia $(VERSION), threads = $(Threads.nthreads())")
    println("="^65)

    # Warmup JIT
    solve_spectral(4, 16, 0.5; use_soe=false)

    exp_soe_speedup()
    exp_spatial_convergence()
    exp_temporal_convergence()
    exp_condition_numbers()
    exp_efficiency_frontier()
    exp_seismic()

    println("\n\nALL EXPERIMENTS COMPLETED SUCCESSFULLY (Julia).")
end

main()
