#!/usr/bin/env julia
using LinearAlgebra, Printf
using SpecialFunctions: gamma

const c_wav = 1.0
const γ_mem = 1.0
const α = 0.5
const T_fin = 1.0
const L_dom = 1.0

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
    return D2, x_asc[2:N]
end

function soe_bases(N_e, Nt)
    s_min = -log(Nt)
    s_max = log(20.0)
    ds = (s_max - s_min) / N_e
    bases = zeros(N_e)
    mu_weights = zeros(N_e)
    coef = (1-α)*sin(π*α)/π * ds
    for i in 1:N_e
        s_l = s_min + (i-0.5)*ds
        mu = exp(s_l)
        bases[i] = exp(-mu * (T_fin/Nt))
        mu_weights[i] = coef * mu^α
    end
    return bases, mu_weights
end

function solve_2d(N_s, Nt)
    A_1D, x_1d = build_chebyshev_D2(N_s)
    n_int = length(x_1d)
    Ndof_2d = n_int^2
    
    I_1d = Matrix(1.0I, n_int, n_int)
    A_2D = kron(A_1D, I_1d) .+ kron(I_1d, A_1D)
    
    A_sc = (T_fin/Nt)^2 * c_wav^2 / 4 .* A_2D
    Lhs = Matrix(1.0I, Ndof_2d, Ndof_2d) .- A_sc
    
    N_exp = 30
    p_loc = 10
    bases, mu_w = soe_bases(N_exp, Nt)
    
    dt = T_fin/Nt
    a0 = dt^(-α)/gamma(2-α)
    gamma_eff = γ_mem * a0
    loc_weights = [(j+1)^(1-α) - j^(1-α) for j in 0:p_loc-1]
    
    exact_u(x, y, t) = t^2 * sin(π * x) * sin(π * y)
    
    X = [x for x in x_1d, y in x_1d][:]
    Y = [y for x in x_1d, y in x_1d][:]
    
    U_curr = exact_u.(X, Y, dt)
    U_prev = zeros(Ndof_2d)
    
    Z = zeros(Ndof_2d, N_exp)
    Hn = zeros(Ndof_2d)
    local_hist = zeros(Ndof_2d, p_loc)
    local_hist[:, 1] .= U_curr .- U_prev
    
    for n in 1:Nt-1
        t_n = n * dt
        Hn .= 0.0
        
        for i in 1:Ndof_2d
            sio = 0.0
            if n > p_loc
                for l in 1:N_exp
                    sio += mu_w[l] * Z[i, l]
                end
            end
            limit = min(n-1, p_loc-1)
            for j in 0:limit
                sio += loc_weights[j+1] * local_hist[i, j+1]
            end
            Hn[i] = sio * gamma_eff
        end
        
        exact_f(x, y, t) = begin
            s = sin(π * x) * sin(π * y)
            utt = 2.0 * s
            lapl = 2.0 * c_wav^2 * (π^2) * (t^2 + dt^2/2) * s
            cap = γ_mem * 2.0 / gamma(3-α) * t^(2-α) * s
            return utt + lapl + cap
        end
        fn = exact_f.(X, Y, t_n)
        
        U_avg = 2 .* U_curr .+ U_prev
        sA = A_sc * U_avg
        rhs = 2 .* U_curr .- U_prev .+ sA .+ (dt^2) .* (fn .- Hn)
        
        U_next = Lhs \ rhs
        
        dU = U_next .- U_curr
        if n > p_loc
            dUpast = local_hist[:, end]
            for i in 1:Ndof_2d
                for l in 1:N_exp
                    Z[i, l] = bases[l] * Z[i, l] + dUpast[i]
                end
            end
        end
        
        for j in p_loc:-1:2
            local_hist[:, j] .= local_hist[:, j-1]
        end
        local_hist[:, 1] .= dU
        
        U_prev .= U_curr
        U_curr .= U_next
    end
    
    exact_end = exact_u.(X, Y, T_fin)
    err = maximum(abs.(U_curr .- exact_end))
    return err
end

function main()
    println("="^60)
    println(" 2D SPATIAL CONVERGENCE (Kronecker Spectral Method)")
    println("="^60)
    Nt = 4096
    Ns_list = [4, 6, 8, 10, 12, 14]
    
    prev_e = NaN
    @printf("%4s   %12s %8s\n", "N", "L_inf Error", "Rate")
    println("-"^40)
    for N in Ns_list
        e = solve_2d(N, Nt)
        
        # Spectral formula: e ~ C exp(-sigma N) => rate = log(e_prev/e)
        # We just print log(error)
        r = isnan(prev_e) ? NaN : log(prev_e/e) / log(N/(N-2))
        fs = isnan(r) ? "  ---" : @sprintf("%5.2f", r)
        @printf("%4d   %12.4e %8s\n", N, e, fs)
        prev_e = e
    end
end

main()
