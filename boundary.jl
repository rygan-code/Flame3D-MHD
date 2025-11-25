function fill_x(Q, U, rankx, ranky)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    # Left Boundary (Inlet side) -> Zero Gradient (Outflow/Transmissive)
    # 也就是 ghost cell 等于内部第一个网格的值
    if rankx == 0 && i <= NG
        # 对应内部网格索引 NG+1 (或者镜像，这里简单的取边界值即可)
        idx_inner = NG + 1
        
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[idx_inner, j, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[idx_inner, j, k, n]
        end
    end

    # Right Boundary (Outlet side) -> Zero Gradient
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

function fill_y(Q, U)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    if j <= NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j+Nyp, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j+Nyp, k, n]
        end
    elseif j > Nyp+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j-Nyp, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j-Nyp, k, n]
        end
    end
    return
end

# Wrapper for Ghost Filling
function fillGhost(Q, U, rankx, ranky)
    # 注意：inlet 参数已被移除，因为 Sod 问题不需要外部入口文件
    @cuda threads=nthreads blocks=nblock fill_x(Q, U, rankx, ranky)
    @cuda threads=nthreads blocks=nblock fill_y(Q, U)
    @cuda threads=nthreads blocks=nblock fill_z(Q, U)
end

# -----------------------------------------------------------------
# 初始化条件：标准 Sod 激波管问题
# 左侧 (x < 0.5): rho=1, p=1, u=0
# 右侧 (x > 0.5): rho=0.125, p=0.1, u=0
# -----------------------------------------------------------------
function init(Q, rankx, ranky, Nprocs)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end
    
    # 只初始化计算域内部，Ghost cell 由 fillGhost 处理
    # 但为了简单起见，全场初始化也没问题，之后第一次迭代前必须调用 fillGhost
    if i >= NG+1 && i <= Nxp+NG && j >= NG+1 && j <= Nyp+NG && k >= NG+1 && k <= Nzp+NG
        
        # 计算全局索引，用于判断隔膜位置
        # 假设总长为 1.0，通过 Grid Index 判断位置
        # Global Index (1-based relative to physical domain start)
        gid_x = (i - NG) + rankx * Nxp 
        total_Nx = Nxp * Nprocs[1]
        
        # 设定隔膜在中间
        is_left = gid_x <= (total_Nx / 2)

            # ρ = 1.0f0
            # p = 1.0f0
            # u = j
            # v = j
            # w = j
        if is_left
            # Left State
            ρ = 1.0f0
            p = 1.0f0
            u = 0.0f0
            v = 0.0f0
            w = 0.0f0
        else
            # Right State
            ρ = 0.125f0
            p = 0.1f0
            u = 0.0f0
            v = 0.0f0
            w = 0.0f0
        end
        
        # 计算温度 T = p / (rho * Rg)
        T = p / (ρ * Rg)

        # 填充 Primitive Variables Q
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
    @cuda threads=nthreads blocks=nblock init(Q, rankx, ranky, Nprocs)
end