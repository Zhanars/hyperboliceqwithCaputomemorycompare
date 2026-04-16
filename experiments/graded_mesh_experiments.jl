#!/usr/bin/env julia
# graded_mesh_experiments.jl — Graded mesh + singular solution experiments (Spectral Only)
# Generates tables for:
# Singular solution u = t^(1+α/2)*sin(πx) degradation + recovery

using LinearAlgebra, Printf
using SpecialFunctions: gamma

const L_dom = 1.0
const T_fin = 1.0
const c_wav = 1.0
const γ_mem = 1.0

# ─────────── CGL nodes and CC weights ───────────
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

# ─────────── Graded temporal mesh ───────────
function graded_mesh(Nt, T, r)
    return [T * (j/Nt)^r for j in 0:Nt]
end

function uniform_mesh(Nt, T)
    return [T * j / Nt for j in 0:Nt]
end

# ─────────── Chebyshev Spectral operator ───────────
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

# ─────────── Spectral solver with arbitrary time grid ───────────
function solve_spectral(A, x_all_nodes, x_int_nodes, w_cc, t_grid, α;
                        exact_u_func, exact_f_frac_func)
    Nt = length(t_grid) - 1
    N_space = length(x_all_nodes) - 1
    n_int = N_space - 1

    # Initialize
    U_prev = zeros(n_int)
    U_curr = [exact_u_func(x_int_nodes[i], t_grid[2]) for i in 1:n_int]

    # Store all time levels for L1 history
    U_hist = zeros(Nt+1, n_int)
    U_hist[1, :] .= U_prev
    U_hist[2, :] .= U_curr

    for n in 1:Nt-1
        tn = t_grid[n+1]
        t_prev = t_grid[n]
        t_next = t_grid[n+2]
        tau_n = t_next - tn
        tau_prev = tn - t_prev

        # L1 Memory at tn
        Hn = zeros(n_int)
        for j in 0:n-1
            dt_j = t_grid[j+2] - t_grid[j+1]
            if n-1-j >= 0
                w_j = ((tn - t_grid[j+1])^(1-α) - (tn - t_grid[j+2])^(1-α)) / gamma(2-α)
                Hn .+= γ_mem * w_j / dt_j .* (U_hist[j+2, :] .- U_hist[j+1, :])
            end
        end

        # Non-uniform 3-point coefficients
        a = 2.0 / (tau_n * (tau_n + tau_prev))
        b = 2.0 / (tau_n * tau_prev)
        c = 2.0 / (tau_prev * (tau_n + tau_prev))

        # True exact continuous RHS: f = U_tt + U(lapl) + U(mem)
        f_compensated = zeros(n_int)
        for i in 1:n_int
            s = sin(π * x_int_nodes[i])
            utt = tn > 0 ? 1.25 * 0.25 * tn^(-0.75) * s : 0.0
            lapl = c_wav^2 * π^2 * tn^1.25 * s
            mem_ex = exact_f_frac_func(x_int_nodes[i], tn)
            f_compensated[i] = utt + lapl + mem_ex
        end

        rhs = b .* U_curr .- c .* U_prev .+ c_wav^2 .* (A * (0.5 .* U_curr .+ 0.25 .* U_prev)) .- Hn .+ f_compensated

        Lhs = Matrix(a*I, n_int, n_int) .- 0.25 .* c_wav^2 .* A
        U_next = Lhs \ rhs

        U_prev = U_curr
        U_curr = U_next
        U_hist[n+2, :] .= U_next
    end

    Uf = zeros(length(w_cc))
    Uf[2:N_space] .= reverse(U_curr)  # map back to ascending order
    e = Uf .- [exact_u_func(x_all_nodes[i], T_fin) for i in eachindex(x_all_nodes)]
    return sqrt(sum(w_cc .* e.^2))
end

# ═══════════════════════════════════════════════════════════
# EXPERIMENT: Singular solution test
# u = t^(1+α/2) * sin(πx) — weak temporal singularity at t=0
# ═══════════════════════════════════════════════════════════
function exp_singular_solution()
    α = 0.5
    β = 1.0 + α/2  # = 1.25 for α=0.5
    N = 16

    println("\n", "="^90)
    println("  SINGULAR SOLUTION: u = t^$β sin(πx)  (α=$α, Spectral N=$N)")
    println("="^90)

    u_exact(x, t) = t^β * sin(π * x)
    caputo_coeff = gamma(β + 1) / gamma(β + 1 - α)
    f_frac(x, t) = γ_mem * caputo_coeff * (t > 0 ? t^(β - α) : 0.0) * sin(π * x)

    r_opt = (2-α)/β
    r_opt = ceil(Int, r_opt) + 1  # Ensures r >= (2-a)/beta

    @printf("\n%6s   %12s %6s   %12s %6s   (Expected: uniform ~%.2f, graded ~%.2f)\n",
            "Nt", "Uniform err", "Rate", "Graded(r=$r_opt) err", "Rate", min(2-α, β), 2-α)
    println("-"^75)

    Nts = [64, 128, 256, 512, 1024, 2048]
    prev_u = nothing
    prev_g = nothing

    D2, x_asc, x_int = build_chebyshev_D2(N)
    w_cc = cc_weights(N)

    for Nt in Nts
        t_uni = uniform_mesh(Nt, T_fin)
        e_uni = solve_spectral(D2, x_asc, x_int, w_cc, t_uni, α;
                               exact_u_func=u_exact, exact_f_frac_func=f_frac)

        t_grd = graded_mesh(Nt, T_fin, r_opt)
        e_grd = solve_spectral(D2, x_asc, x_int, w_cc, t_grd, α;
                               exact_u_func=u_exact, exact_f_frac_func=f_frac)

        r_u = isnothing(prev_u) ? NaN : log(prev_u/e_uni) / log(2)
        
        # Rate for graded mesh uses max step size ratio. Actually for Nt/2, Ntt_max ratio is approx 2^r.
        # A fairer rate calculation against N is exactly log2. 
        # Since r_opt = 2, max step scales as (1/N)^2. But error bound is O(N_t^{-(2-a)}).
        # So we just use log2 ratio against Nt.
        r_g = isnothing(prev_g) ? NaN : log(prev_g/e_grd) / log(2)

        fu(r) = isnan(r) ? "  ---" : @sprintf("%5.2f", r)

        @printf("%6d   %12.4e %6s   %12.4e %6s\n",
                Nt, e_uni, fu(r_u), e_grd, fu(r_g))

        prev_u = e_uni
        prev_g = e_grd
    end
end

function main()
    println("Julia $(VERSION)")
    exp_singular_solution()
end

main()
