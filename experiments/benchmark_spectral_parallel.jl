#!/usr/bin/env julia
using LinearAlgebra, Printf, Base.Threads
using SpecialFunctions: gamma

const L_dom = 1.0
const T_fin = 1.0
const c_wav = 1.0
const γ_mem = 1.0
const α = 0.5
const N_space = 256  # Size of spatial grid to make matrix multiplications prominent
const Nt = 1024     # Large number of time steps to emphasize the memory history sum

# SOE parameters
const N_exp = 40
const p_loc = 20

# Chebyshev setup
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
    return D2
end

function soe_bases(N_e)
    # Roots and weights for SOE approximation
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

function benchmark_run()
    n_int = N_space - 1
    A = build_chebyshev_D2(N_space)
    A_sc = (T_fin/Nt)^2 * c_wav^2 / 4 .* A
    Lhs = Matrix(1.0I, n_int, n_int) .- A_sc
    
    bases, mu_w = soe_bases(N_exp)
    
    U_curr = zeros(n_int)
    U_prev = zeros(n_int)
    
    Z = zeros(n_int, N_exp)
    Hn = zeros(n_int)
    local_hist = zeros(n_int, p_loc)
    
    dt = T_fin/Nt
    a0 = dt^(-α)/gamma(2-α)
    gamma_eff = γ_mem * a0
    loc_weights = [(j+1)^(1-α) - j^(1-α) for j in 0:p_loc-1]
    
    # Pre-allocate for threads
    T = Threads.nthreads()
    
    # Time loop
    for n in 1:Nt-1
        Hn .= 0.0
        
        # Parallel memory update
        Threads.@threads for i in 1:n_int
            sio = 0.0
            # SOE tail
            if n > p_loc
                for l in 1:N_exp
                    sio += mu_w[l] * Z[i, l]
                end
            end
            
            # Local exact
            limit = min(n-1, p_loc-1)
            for j in 0:limit
                sio += loc_weights[j+1] * local_hist[i, j+1]
            end
            Hn[i] = sio * gamma_eff
        end
        
        # RHS Matrix-Vector & assembly
        rhs = 2 .* U_curr .- U_prev .- Hn
        Threads.@threads for i in 1:n_int
            sA = 0.0
            for j in 1:n_int
                sA += A_sc[i, j] * (2*U_curr[j] + U_prev[j])
            end
            rhs[i] += sA + 0.01 # fake forcing
        end
        
        U_next = Lhs \ rhs
        
        # Update states
        dU = U_next .- U_curr
        if n > p_loc
            dUpast = local_hist[:, end]
            Threads.@threads for i in 1:n_int
                for l in 1:N_exp
                    Z[i, l] = bases[l] * Z[i, l] + dUpast[i]
                end
            end
        end
        
        # Shift local window
        for j in p_loc:-1:2
            local_hist[:, j] .= local_hist[:, j-1]
        end
        local_hist[:, 1] .= dU
        
        U_prev .= U_curr
        U_curr .= U_next
    end
    return U_curr[1]
end

# Warmup JIT
benchmark_run()

# Measure
t_measure = @elapsed benchmark_run()

open("benchmark_times.txt", "a") do io
    println(io, "$(Threads.nthreads()),$t_measure")
end
