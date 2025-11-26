function fill_x(Q, U, rankx, ranky)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    # --- Inlet (Left Boundary) ---
    # 强制固定为来流条件 (Dirichlet)
    if rankx == 0 && i <= NG
        # 自由来流参数 (Freestream) - 请根据需要修改
        ρ_inf = 1.0f0
        u_inf = 0.5f0  # 马赫数约为 0.5 (假设 c=1)
        v_inf = 0.0f0
        w_inf = 0.0f0
        p_inf = 1.0f0 / 1.4f0 # 使得声速 c = 1.0
        T_inf = p_inf / (ρ_inf * Rg)

        for n = 1:Nprim
            if n==1; val = ρ_inf;
            elseif n==2; val = u_inf;
            elseif n==3; val = v_inf;
            elseif n==4; val = w_inf;
            elseif n==5; val = p_inf;
            elseif n==6; val = T_inf; end
            
            @inbounds Q[i, j, k, n] = val
        end
        
        # 更新守恒变量 U
        @inbounds U[i, j, k, 1] = ρ_inf
        @inbounds U[i, j, k, 2] = ρ_inf * u_inf
        @inbounds U[i, j, k, 3] = ρ_inf * v_inf
        @inbounds U[i, j, k, 4] = ρ_inf * w_inf
        @inbounds U[i, j, k, 5] = p_inf/(γ-1) + 0.5f0*ρ_inf*(u_inf^2 + v_inf^2 + w_inf^2)
    end

    # --- Outlet (Right Boundary) ---
    # 零梯度外推 (Zero Gradient)
    if rankx == (Nprocs[1]-1) && i > Nxp+NG
        idx_inner = Nxp + NG
        
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[idx_inner, j, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[idx_inner, j, k, n]
        end
    end
    return
end

# Y 方向：壁面 (Wall) 和 远场 (Farfield)
function fill_y(Q, U, ranky)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    # --- Bottom Wall (下壁面，平板) ---
    # 无滑移绝热壁面 (No-Slip Adiabatic Wall)
    if ranky == 0 && j <= NG
        # 镜像索引: j=NG 对应 j=NG+1 (inner)
        idx_inner = 2*NG + 1 - j
        
        # 密度 & 压力：零梯度 (绝热壁面 dp/dn = 0, dT/dn = 0 => d_rho/dn = 0)
        @inbounds Q[i, j, k, 1] = Q[i, idx_inner, k, 1] # rho
        @inbounds Q[i, j, k, 5] = Q[i, idx_inner, k, 5] # p
        @inbounds Q[i, j, k, 6] = Q[i, idx_inner, k, 6] # T (绝热)
        
        # 速度：反向 (No-slip condition: u_wall = 0 => u_ghost = -u_inner)
        @inbounds Q[i, j, k, 2] = -Q[i, idx_inner, k, 2] # u
        @inbounds Q[i, j, k, 3] = -Q[i, idx_inner, k, 3] # v (法向也反向)
        @inbounds Q[i, j, k, 4] = -Q[i, idx_inner, k, 4] # w

        # 更新 U
        ρ = Q[i, j, k, 1]
        u = Q[i, j, k, 2]
        v = Q[i, j, k, 3]
        w = Q[i, j, k, 4]
        p = Q[i, j, k, 5]
        
        @inbounds U[i, j, k, 1] = ρ
        @inbounds U[i, j, k, 2] = ρ * u
        @inbounds U[i, j, k, 3] = ρ * v
        @inbounds U[i, j, k, 4] = ρ * w
        @inbounds U[i, j, k, 5] = p/(γ-1) + 0.5f0*ρ*(u^2+v^2+w^2)
    end

    # --- Top Boundary (上边界，远场) ---
    # 设为远场边界 (Farfield/Freestream) 或者 简单的零梯度/滑移
    # 这里使用：固定为自由来流 (Dirichlet)，假设上边界足够高
    if ranky == (Nprocs[2]-1) && j > Nyp+NG
        ρ_inf = 1.0f0
        u_inf = 0.5f0
        v_inf = 0.0f0
        w_inf = 0.0f0
        p_inf = 1.0f0 / 1.4f0
        T_inf = p_inf / (ρ_inf * Rg)

        @inbounds Q[i, j, k, 1] = ρ_inf
        @inbounds Q[i, j, k, 2] = u_inf
        @inbounds Q[i, j, k, 3] = v_inf
        @inbounds Q[i, j, k, 4] = w_inf
        @inbounds Q[i, j, k, 5] = p_inf
        @inbounds Q[i, j, k, 6] = T_inf

        @inbounds U[i, j, k, 1] = ρ_inf
        @inbounds U[i, j, k, 2] = ρ_inf * u_inf
        @inbounds U[i, j, k, 3] = ρ_inf * v_inf
        @inbounds U[i, j, k, 4] = ρ_inf * w_inf
        @inbounds U[i, j, k, 5] = p_inf/(γ-1) + 0.5f0*ρ_inf*(u_inf^2+v_inf^2+w_inf^2)
    end
    return
end

# Z 方向：保持周期性 (Periodic)
function fill_z(Q, U)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    if k <= NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, k+Nzp, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k+Nzp, n]
        end
    elseif k > Nzp+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, k-Nzp, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k-Nzp, n]
        end
    end
    return
end

# Wrapper for Ghost Filling
function fillGhost(Q, U, rankx, ranky)
    # 注意：inlet 参数已被移除，因为 Sod 问题不需要外部入口文件
    @cuda threads=nthreads blocks=nblock fill_x(Q, U, rankx, ranky)
    @cuda threads=nthreads blocks=nblock fill_y(Q, U, ranky)
    @cuda threads=nthreads blocks=nblock fill_z(Q, U)
end

# -----------------------------------------------------------------
# 初始化条件：标准 Sod 激波管问题
# 左侧 (x < 0.5): rho=1, p=1, u=0
# 右侧 (x > 0.5): rho=0.125, p=0.1, u=0
# -----------------------------------------------------------------
function init(Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end
    
    # 只初始化内部区域，Ghost 由 fillGhost 处理
    if i >= NG+1 && i <= Nxp+NG && j >= NG+1 && j <= Nyp+NG && k >= NG+1 && k <= Nzp+NG
        
        # 自由来流参数 (Freestream Parameters)
        # 建议：Ma = 0.5, Re 由粘性系数 mu 控制
        ρ = 1.0f0
        u = 0.5f0
        v = 0.0f0
        w = 0.0f0
        p = 1.0f0 / 1.4f0
        T = p / (ρ * Rg)

        # 填充 Primitive
        @inbounds Q[i, j, k, 1] = ρ
        @inbounds Q[i, j, k, 2] = u
        @inbounds Q[i, j, k, 3] = v
        @inbounds Q[i, j, k, 4] = w
        @inbounds Q[i, j, k, 5] = p
        @inbounds Q[i, j, k, 6] = T
    end
    return
end

# Initialization wrapper
# 注意：这里我添加了 U, rankx, Nprocs 到参数列表
function initialize(Q, rankx, ranky, Nprocs)
    @cuda threads=nthreads blocks=nblock init(Q)
end