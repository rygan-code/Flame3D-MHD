using MPI
using StaticArrays, CUDA
using CUDA:i32
using HDF5, DelimitedFiles
using Dates, Printf

CUDA.allowscalar(false)

include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")
include("mpi.jl")
include("IO.jl")
include("FVM.jl")

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, s1, s2, s3, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, x, y, z, ϕ)

    if finite_volume
        if eigen_reconstruction
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Eigen_reconstruct_i(Q, U, ϕ, s1, Fx, x, y, z)
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Eigen_reconstruct_j(Q, U, ϕ, s2, Fy, x, y, z)
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Eigen_reconstruct_k(Q, U, ϕ, s3, Fz, x, y, z)
        else
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Conser_reconstruct_i(Q, U, ϕ, s1, Fx, x, y, z)
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Conser_reconstruct_j(Q, U, ϕ, s2, Fy, x, y, z)
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Conser_reconstruct_k(Q, U, ϕ, s3, Fz, x, y, z)
        end
        # Fx_cpu_FVM = Array(Fx)
        # @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
        # @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_xc(Fx, ϕ, s1, Fp, Fm, Q, dξdx, dξdy, dξdz)
        # # @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
        # # @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_yc(Fy, ϕ, s2, Fp, Fm, Q, dηdx, dηdy, dηdz)
        # Fx_cpu_FDM = Array(Fx)
        # @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Conser_reconstruct_j(Q, U, ϕ, s2, Fy, dηdx, dηdy, dηdz)
        # for i = 1:Nxp+1
        #     @printf("i = %d, Fvm1 = %f, Fvm2 = %f, Fvm3 = %f, Fvm4 = %f, Fvm5 = %f, Fdm1 = %f, Fdm2 = %f, Fdm3 = %f, Fdm4 = %f, Fdm5 = %f\n", i, Fx_cpu_FVM[i, 10, 10, 1], Fx_cpu_FVM[i, 10, 10, 2], Fx_cpu_FVM[i, 10, 10, 3], Fx_cpu_FVM[i, 10, 10, 4], Fx_cpu_FVM[i, 10, 10, 5], Fx_cpu_FDM[i, 10, 10, 1], Fx_cpu_FDM[i, 10, 10, 2], Fx_cpu_FDM[i, 10, 10, 3], Fx_cpu_FDM[i, 10, 10, 4], Fx_cpu_FDM[i, 10, 10, 5])
        # end
        # exit(1)
            

        # 1. 使用 CUDA 高效函数检查是否存在 NaN
        # any(isnan, Array) 是 GPU 上检查 NaN 的标准写法，无需分配额外内存
        # has_nan_x = any(isnan, Fx)
        # has_nan_y = any(isnan, Fy)
        # has_nan_z = any(isnan, Fz)

        # if has_nan_x || has_nan_y || has_nan_z
        #     println("\n========================================")
        #     println("🔴 CRITICAL ERROR: NaN detected in fluxes!")
        #     println("========================================")
            
        #     println("Downloading arrays to CPU for debugging...")
            
        #     # 将显存数据拷贝到内存
        #     Fx_cpu = Array(Fx)
        #     Fy_cpu = Array(Fy)
        #     Fz_cpu = Array(Fz)
            
        #     max_report_count = 10 # 限制打印数量，防止刷屏

        #     # --- 扫描 X 方向通量 ---
        #     if has_nan_x
        #         println("\n🔎 Scanning Fx (X-Fluxes)...")
        #         count = 0
        #         # 获取维度: 假设维度顺序是 [i, j, k, n]
        #         Nx, Ny, Nz, Nv = size(Fx_cpu)
                
        #         for k = 1:Nz, j = 1:Ny, i = 1:Nx
        #             # 检查该点 5 个变量中是否有任意一个是 NaN
        #             if any(isnan, @view Fx_cpu[i, j, k, :])
        #                 # 找出具体是第几个变量坏了
        #                 bad_vars = findall(isnan, @view Fx_cpu[i, j, k, :])
        #                 @printf("   [Fx] NaN found at (i=%d, j=%d, k=%d), Variables: %s\n", i, j, k, string(bad_vars))
                        
        #                 # 顺便打印该点的数值，方便分析
        #                 # println("        Values: ", Fx_cpu[i, j, k, :]) 
                        
        #                 count += 1
        #                 if count >= max_report_count
        #                     println("   ... (Stopped reporting Fx errors, too many NaNs)")
        #                     break
        #                 end
        #             end
        #         end
        #     end

        #     # --- 扫描 Y 方向通量 ---
        #     if has_nan_y
        #         println("\n🔎 Scanning Fy (Y-Fluxes)...")
        #         count = 0
        #         Nx, Ny, Nz, Nv = size(Fy_cpu)
                
        #         for k = 1:Nz, j = 1:Ny, i = 1:Nx
        #             if any(isnan, @view Fy_cpu[i, j, k, :])
        #                 bad_vars = findall(isnan, @view Fy_cpu[i, j, k, :])
        #                 @printf("   [Fy] NaN found at (i=%d, j=%d, k=%d), Variables: %s\n", i, j, k, string(bad_vars))
                        
        #                 count += 1
        #                 if count >= max_report_count
        #                     println("   ... (Stopped reporting Fy errors)")
        #                     break
        #                 end
        #             end
        #         end
        #     end

        #     # --- 扫描 Z 方向通量 ---
        #     if has_nan_z
        #         println("\n🔎 Scanning Fz (Z-Fluxes)...")
        #         count = 0
        #         Nx, Ny, Nz, Nv = size(Fz_cpu)
                
        #         for k = 1:Nz, j = 1:Ny, i = 1:Nx
        #             if any(isnan, @view Fz_cpu[i, j, k, :])
        #                 bad_vars = findall(isnan, @view Fz_cpu[i, j, k, :])
        #                 @printf("   [Fz] NaN found at (i=%d, j=%d, k=%d), Variables: %s\n", i, j, k, string(bad_vars))
                        
        #                 count += 1
        #                 if count >= max_report_count
        #                     println("   ... (Stopped reporting Fz errors)")
        #                     break
        #                 end
        #             end
        #         end
        #     end
            
        #     println("\nExiting due to numerical instability.")
        #     exit(1)
        # end
    else
        if splitMethod == "SW"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
        elseif splitMethod == "LF"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_LF(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
        elseif splitMethod == "VL"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_VL(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
        elseif splitMethod == "AUSM"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_AUSM(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
        else
            error("Not valid split method")
        end
        if character
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_xc(Fx, ϕ, s1, Fp, Fm, Q, dξdx, dξdy, dξdz)
        else
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_x(Fx, ϕ, s1, Fp, Fm, Ncons)
        end

        if splitMethod == "SW"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
        elseif splitMethod == "LF"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_LF(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
        elseif splitMethod == "VL"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_VL(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
        elseif splitMethod == "AUSM"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_AUSM(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
        else
            error("Not valid split method")
        end
        if character
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_yc(Fy, ϕ, s2, Fp, Fm, Q, dηdx, dηdy, dηdz)
        else
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_y(Fy, ϕ, s2, Fp, Fm, Ncons)
        end

        if splitMethod == "SW"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s3, dζdx, dζdy, dζdz)
        elseif splitMethod == "LF"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_LF(Q, Fp, Fm, s3, dζdx, dζdy, dζdz)
        elseif splitMethod == "VL"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_VL(Q, Fp, Fm, s3, dζdx, dζdy, dζdz)
        elseif splitMethod == "AUSM"
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_AUSM(Q, Fp, Fm, s3, dζdx, dζdy, dζdz)
        else
            error("Not valid split method")
        end
        if character
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_zc(Fz, ϕ, s3, Fp, Fm, Q, dζdx, dζdy, dζdz)
        else
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_z(Fz, ϕ, s3, Fp, Fm, Ncons)
        end
    end

    if viscous
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux(Fv_x, Fv_y, Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)
    end
end

function time_step(rank, comm_cart)
    Nx_tot = Nxp+2*NG
    Ny_tot = Nyp+2*NG
    Nz_tot = Nzp+2*NG

    # global indices
    (rankx, ranky, rankz) = MPI.Cart_coords(comm_cart, rank)

    lox = rankx*Nxp+1
    hix = (rankx+1)*Nxp+2*NG
    hix_without_ghost = (rankx+1)*Nxp

    loy = ranky*Nyp+1
    hiy = (ranky+1)*Nyp+2*NG
    hiy_without_ghost = (ranky+1)*Nyp

    loz = rankz*Nzp+1
    hiz = (rankz+1)*Nzp+2*NG
    hiz_without_ghost = (rankz+1)*Nzp

    if restart[end-2:end] == ".h5"
        if rank == 0
            printstyled("Restart\n", color=:yellow)
        end
        fid = h5open(restart, "r", comm_cart)
        Q_h = fid["Q_h"][lox:hix, loy:hiy, loz:hiz, :, 1]
        close(fid)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat", Float32)

        Q = cu(Q_h)
        inlet = cu(inlet_h)
    else
        Q_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
        Q = CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat", Float32)

        copyto!(Q_h, Q)
        inlet = cu(inlet_h)

        initialize(Q, rankx, ranky, Nprocs)
    end
    
    ϕ_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # shock sensor

    # load mesh metrics
    fid = h5open(metrics, "r", comm_cart)
    dξdx_h = fid["dξdx"][lox:hix, loy:hiy, loz:hiz]
    dξdy_h = fid["dξdy"][lox:hix, loy:hiy, loz:hiz]
    dξdz_h = fid["dξdz"][lox:hix, loy:hiy, loz:hiz]
    dηdx_h = fid["dηdx"][lox:hix, loy:hiy, loz:hiz]
    dηdy_h = fid["dηdy"][lox:hix, loy:hiy, loz:hiz]
    dηdz_h = fid["dηdz"][lox:hix, loy:hiy, loz:hiz]
    dζdx_h = fid["dζdx"][lox:hix, loy:hiy, loz:hiz]
    dζdy_h = fid["dζdy"][lox:hix, loy:hiy, loz:hiz]
    dζdz_h = fid["dζdz"][lox:hix, loy:hiy, loz:hiz] 

    J_h = fid["J"][lox:hix, loy:hiy, loz:hiz] 
    close(fid)

    # load mesh coordinates
    x_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
    y_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
    z_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
    fid = h5open(mesh, "r", comm_cart)
    ix_g_start = max(1,  lox - NG) ; ix_l_start = ix_g_start - lox + NG + 1
    ix_g_end   = min(Nx, hix - NG) ; ix_l_end   = ix_g_end   - lox + NG + 1

    iy_g_start = max(1,  loy - NG) ; iy_l_start = iy_g_start - loy + NG + 1
    iy_g_end   = min(Ny, hiy - NG) ; iy_l_end   = iy_g_end   - loy + NG + 1
    
    iz_g_start = max(1,  loz - NG) ; iz_l_start = iz_g_start - loz + NG + 1
    iz_g_end   = min(Nz, hiz - NG) ; iz_l_end   = iz_g_end   - loz + NG + 1
    
    if (ix_g_end >= ix_g_start) && (iy_g_end >= iy_g_start) && (iz_g_end >= iz_g_start)
        x_h[ix_l_start:ix_l_end, iy_l_start:iy_l_end, iz_l_start:iz_l_end] = fid["coords"][1, ix_g_start:ix_g_end, iy_g_start:iy_g_end, iz_g_start:iz_g_end]
        y_h[ix_l_start:ix_l_end, iy_l_start:iy_l_end, iz_l_start:iz_l_end] = fid["coords"][2, ix_g_start:ix_g_end, iy_g_start:iy_g_end, iz_g_start:iz_g_end]
        z_h[ix_l_start:ix_l_end, iy_l_start:iy_l_end, iz_l_start:iz_l_end] = fid["coords"][3, ix_g_start:ix_g_end, iy_g_start:iy_g_end, iz_g_start:iz_g_end]
    end
    close(fid)

    extrapolation(x_h, y_h, z_h, rankx, ranky, rankz)

    # move to device memory
    dξdx = cu(dξdx_h)
    dξdy = cu(dξdy_h)
    dξdz = cu(dξdz_h)
    dηdx = cu(dηdx_h)
    dηdy = cu(dηdy_h)
    dηdz = cu(dηdz_h)
    dζdx = cu(dζdx_h)
    dζdy = cu(dζdy_h)
    dζdz = cu(dζdz_h)
    J = cu(J_h)
    s1 = @. sqrt(dξdx^2+dξdy^2+dξdz^2)
    s2 = @. sqrt(dηdx^2+dηdy^2+dηdz^2)
    s3 = @. sqrt(dζdx^2+dζdy^2+dζdz^2)
    x = cu(x_h)
    y = cu(y_h)
    z = cu(z_h)

    # allocate on device
    ϕ  =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # Shock sensor
    U  =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fp =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   CUDA.zeros(Float32, Nxp+1, Nyp, Nzp, Ncons)
    Fy =   CUDA.zeros(Float32, Nxp, Nyp+1, Nzp, Ncons)
    Fz =   CUDA.zeros(Float32, Nxp, Nyp, Nzp+1, Ncons)
    Fv_x = CUDA.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)
    Fv_y = CUDA.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)
    Fv_z = CUDA.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)
    if LTS
        LTS_dt = CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
    end

    Un = similar(U)

    if average
        Q_avg = CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
    end

    if filtering && filtering_nonlinear
        sc = CUDA.zeros(Float32, Nxp, Nyp, Nzp)
    end

    # MPI buffer 
    Qsbuf_hx = zeros(Float32, NG, Ny_tot, Nz_tot, Nprim)
    Qsbuf_hy = zeros(Float32, Nx_tot, NG, Nz_tot, Nprim)
    Qsbuf_hz = zeros(Float32, Nx_tot, Ny_tot, NG, Nprim)
    Qrbuf_hx = similar(Qsbuf_hx)
    Qrbuf_hy = similar(Qsbuf_hy)
    Qrbuf_hz = similar(Qsbuf_hz)
    Mem.pin(Qsbuf_hx)
    Mem.pin(Qsbuf_hy)
    Mem.pin(Qsbuf_hz)
    Mem.pin(Qrbuf_hx)
    Mem.pin(Qrbuf_hy)
    Mem.pin(Qrbuf_hz)

    Qsbuf_dx = cu(Qsbuf_hx)
    Qsbuf_dy = cu(Qsbuf_hy)
    Qsbuf_dz = cu(Qsbuf_hz)
    Qrbuf_dx = cu(Qrbuf_hx)
    Qrbuf_dy = cu(Qrbuf_hy)
    Qrbuf_dz = cu(Qrbuf_hz)

    # initial
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock prim2c(U, Q)
    exchange_ghost(Q, Nprim, comm_cart, 
                   Qsbuf_hx, Qsbuf_dx, Qrbuf_hx, Qrbuf_dx,
                   Qsbuf_hy, Qsbuf_dy, Qrbuf_hy, Qrbuf_dy,
                   Qsbuf_hz, Qsbuf_dz, Qrbuf_hz, Qrbuf_dz)
    fillGhost(Q, U, rankx, ranky)

    # sampling metadata
    if sample
        sample_count::Int64 = 1
        valid_rankx = -1
        valid_ranky = -1
        valid_rankz = -1
    
        # find target ranks
        if sample_index[1] ≠ -1
            local_rankx::Int64 = (sample_index[1]-1) ÷ Nxp
            local_idx::Int64 = (sample_index[1]-1) % Nxp + 1
    
            if rankx == local_rankx
                valid_rankx = rank
            end
    
            # collect on rank 0
            if rank == 0
                collectionx = zeros(Float32, Ny, Nz, Nprim)
                rank_listx = MPI.Gather(valid_rankx, comm_cart)
                rank_listx = filter!(x->x!=-1, rank_listx)
            else
                MPI.Gather(valid_rankx, comm_cart)
            end
        end
    
        if sample_index[2] ≠ -1
            local_ranky::Int64 = (sample_index[2]-1) ÷ Nyp
            local_idy::Int64 = (sample_index[2]-1) % Nyp + 1
    
            if ranky == local_ranky
                valid_ranky = rank
            end
    
            # collect on rank 0
            if rank == 0
                collectiony = zeros(Float32, Nx, Nz, Nprim)
                rank_listy = MPI.Gather(valid_ranky, comm_cart)
                rank_listy = filter!(x->x!=-1, rank_listy)
            else
                MPI.Gather(valid_ranky, comm_cart)
            end
        end
    
        if sample_index[3] ≠ -1
            local_rankz::Int64 = (sample_index[3]-1) ÷ Nzp
            local_idz::Int64 = (sample_index[3]-1) % Nzp + 1
    
            if rankz == local_rankz
                valid_rankz = rank
            end
    
            # collect on rank 0
            if rank == 0
                collectionz = zeros(Float32, Nx, Ny, Nprim)
                rank_listz = MPI.Gather(valid_rankz, comm_cart)
                rank_listz = filter!(x->x!=-1, rank_listz)
            else
                MPI.Gather(valid_rankz, comm_cart)
            end
        end
    end

    for tt = 1:ceil(Int, Time/dt)
        if tt*dt > Time || tt > maxStep
            return
        end

        # RK3
        for KRK = 1:3
            if KRK == 1
                copyto!(Un, U)
                if LTS
                    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock compute_dt(LTS_dt, Q, J, s1, s2, s3)
                end
            end

            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q)
            flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, s1, s2, s3, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, x, y, z, ϕ)
            if LTS
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock div_LTS(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, LTS_dt, J)
            else
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
            end

            if KRK == 2
                @cuda maxregs=maxreg fastmath=true threads=nthreads2 blocks=nblock2 linComb(U, Un, Ncons, 0.25f0, 0.75f0)
            elseif KRK == 3
                @cuda maxregs=maxreg fastmath=true threads=nthreads2 blocks=nblock2 linComb(U, Un, Ncons, 2/3f0, 1/3f0)
            end

            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock c2Prim(U, Q)
            exchange_ghost(Q, Nprim, comm_cart, 
                           Qsbuf_hx, Qsbuf_dx, Qrbuf_hx, Qrbuf_dx,
                           Qsbuf_hy, Qsbuf_dy, Qrbuf_hy, Qrbuf_dy,
                           Qsbuf_hz, Qsbuf_dz, Qrbuf_hz, Qrbuf_dz)
            fillGhost(Q, U, rankx, ranky)
        end

        if filtering && tt % filtering_interval == 0
            copyto!(Un, U)
            if filtering_nonlinear
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock pre_x(Q, sc, filtering_rth)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock filter_x(U, Un, sc, filtering_s0)
            else
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linearFilter_x(U, Un, filtering_s0)
            end

            copyto!(Un, U)
            if filtering_nonlinear
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock pre_y(Q, sc, filtering_rth)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock filter_y(U, Un, sc, filtering_s0)
            else
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linearFilter_y(U, Un, filtering_s0)
            end

            copyto!(Un, U)
            if filtering_nonlinear
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock pre_z(Q, sc, filtering_rth)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock filter_z(U, Un, sc, filtering_s0)
            else
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linearFilter_z(U, Un, filtering_s0)
            end
        end

        if tt % 10 == 0 && rank == 0
            printstyled("Step: ", color=:cyan)
            @printf "%g" tt
            printstyled("\tTime: ", color=:blue)
            @printf "%.2e" tt*dt
            printstyled("\tWall time: ", color=:green)
            println("$(now())")
            flush(stdout)

            if any(isnan, U)
                printstyled("Oops, NaN detected\n", color=:red)
                flush(stdout)
                MPI.Abort(comm_cart, 1)
                return
            end
        end

        if plt_xdmf
            plotFile_xdmf(tt, Q, ϕ, Q_h, ϕ_h, comm_cart, rank, rankx, ranky, rankz)
        else
            plotFile_h5(tt, Q, Q_h, comm_cart, rank, rankx, ranky, rankz)
        end

        checkpointFile(tt, Q_h, Q, comm_cart, rank)

        # Average output
        if average && tt <= avg_step*avg_total
            if tt % avg_step == 0
                @. Q_avg += Q/avg_total
            end

            if tt == avg_step*avg_total
                if rank == 0
                    printstyled("average done\n", color=:green)
                    mkpath("./AVG")
                end

                averageFile(tt, Q_avg, Q_h, comm_cart, rankx, ranky, rankz)                
            end
        end

        # collection of slice
        if sample && (tt % sample_step == 0)

            if rank == 0
                mkpath("./SAMPLE")
            end

            if sample_index[1] ≠ -1
                if rankx == local_rankx && rank ≠ 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[local_idx, 1+NG:Nyp+NG, 1+NG:Nzp+NG, :]
                    MPI.Send(part, 0, 0, comm_cart)
                end
            
                if rank == 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[local_idx, 1+NG:Nyp+NG, 1+NG:Nzp+NG, :]
                    for i ∈ rank_listx
                        if i ≠ 0
                            MPI.Recv!(part, i, 0, comm_cart)
                        end
            
                        # get global index
                        (_, ry, rz) = MPI.Cart_coords(comm_cart, i)
            
                        ly = ry*Nyp+1
                        hy = (ry+1)*Nyp
            
                        lz = rz*Nzp+1
                        hz = (rz+1)*Nzp
            
                        collectionx[ly:hy, lz:hz, :] = part
                    end

                    # write and append HDF5 dataset
                    if sample_count == 1
                        h5open("./SAMPLE/collection-x.h5", "w") do file
                            dset = create_dataset(
                                file,
                                "collection",
                                datatype(Float32),
                                dataspace((Ny, Nz, Nprim, 1), (-1,-1,-1,-1));
                                chunk=(Ny, Nz, Nprim, 1),
                                shuffle=plt_shuffle,
                                compress=plt_compress_level
                            )
                            dset[:, :, :, 1] = collectionx
                        end
                    else
                        h5open("./SAMPLE/collection-x.h5", "r+") do file
                            dset = file["collection"]
                            HDF5.set_extent_dims(dset, (Ny, Nz, Nprim, sample_count))
                            dset[:, :, :, end] = collectionx
                        end
                    end
                end
            end

            if sample_index[2] ≠ -1
                if ranky == local_ranky && rank ≠ 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, local_idy, 1+NG:Nzp+NG, :]
                    MPI.Send(part, 0, 0, comm_cart)
                end
            
                if rank == 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, local_idy, 1+NG:Nzp+NG, :]
                    for i ∈ rank_listy
                        if i ≠ 0
                            MPI.Recv!(part, i, 0, comm_cart)
                        end

                        # get global index
                        (rx, _, rz) = MPI.Cart_coords(comm_cart, i)
            
                        lx = rx*Nxp+1
                        hx = (rx+1)*Nxp
            
                        lz = rz*Nzp+1
                        hz = (rz+1)*Nzp
            
                        collectiony[lx:hx, lz:hz, :] = part
                    end

                    if sample_count == 1
                        h5open("./SAMPLE/collection-y.h5", "w") do file
                            dset = create_dataset(
                                file,
                                "collection",
                                datatype(Float32),
                                dataspace((Nx, Nz, Nprim, 1), (-1,-1,-1,-1));
                                chunk=(Nx, Nz, Nprim, 1),
                                shuffle=plt_shuffle,
                                compress=plt_compress_level
                            )
                            dset[:, :, :, 1] = collectiony
                        end
                    else
                        h5open("./SAMPLE/collection-y.h5", "r+") do file
                            dset = file["collection"]
                            HDF5.set_extent_dims(dset, (Nx, Nz, Nprim, sample_count))
                            dset[:, :, :, end] = collectiony
                        end
                    end
                end
            end
            
            if sample_index[3] ≠ -1
                if rankz == local_rankz && rank ≠ 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, local_idz, :]
                    MPI.Send(part, 0, 0, comm_cart)
                end
            
                if rank == 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, local_idz, :]
                    for i ∈ rank_listz
                        if i ≠ 0
                            MPI.Recv!(part, i, 0, comm_cart)
                        end

                        # get global index
                        (rx, ry, _) = MPI.Cart_coords(comm_cart, i)
            
                        lx = rx*Nxp+1
                        hx = (rx+1)*Nxp
            
                        ly = ry*Nyp+1
                        hy = (ry+1)*Nyp
            
                        collectionz[lx:hx, ly:hy, :] = part
                    end

                    if sample_count == 1
                        h5open("./SAMPLE/collection-z.h5", "w") do file
                            dset = create_dataset(
                                file,
                                "collection",
                                datatype(Float32),
                                dataspace((Nx, Ny, Nprim, 1), (-1,-1,-1,-1));
                                chunk=(Nx, Ny, Nprim, 1),
                                shuffle=plt_shuffle,
                                compress=plt_compress_level
                            )
                            dset[:, :, :, 1] = collectionz
                        end
                    else
                        h5open("./SAMPLE/collection-z.h5", "r+") do file
                            dset = file["collection"]
                            HDF5.set_extent_dims(dset, (Nx, Ny, Nprim, sample_count))
                            dset[:, :, :, end] = collectionz
                        end
                    end
                end
            end

            sample_count += 1
        end
    end
    if rank == 0
        printstyled("Done!\n", color=:green)
        flush(stdout)
    end
    MPI.Barrier(comm_cart)
    return
end

function extrapolation(x, y, z, rankx, ranky, rankz)
    # 局部网格参数 (假设 Nxp, Nyp, Nzp, NG, Nprocs 为全局常量)
    # 如果不是常量，请作为参数传入
    
    # 内部计算域范围
    I_in = NG+1 : Nxp+NG
    J_in = NG+1 : Nyp+NG
    K_in = NG+1 : Nzp+NG
    
    # Ghost 范围
    I_gL = 1:NG
    I_gR = Nxp+NG+1 : Nxp+2*NG
    J_gL = 1:NG
    J_gR = Nyp+NG+1 : Nyp+2*NG
    K_gL = 1:NG
    K_gR = Nzp+NG+1 : Nzp+2*NG

    # ==========================================
    # 1. X Direction Faces (Left & Right)
    # ==========================================
    
    # Left Boundary (Physical)
    if rankx == 0
        @inbounds for k in K_in, j in J_in, i in I_gL
            # x[i] = 2*x[NG+1] - x[2*NG+2-i]
            idx_b   = NG + 1
            idx_src = 2*NG + 2 - i
            x[i, j, k] = 2*x[idx_b, j, k] - x[idx_src, j, k]
            y[i, j, k] = 2*y[idx_b, j, k] - y[idx_src, j, k]
            z[i, j, k] = 2*z[idx_b, j, k] - z[idx_src, j, k]
        end
    end

    # Right Boundary (Physical)
    if rankx == Nprocs[1] - 1
        @inbounds for k in K_in, j in J_in, i in I_gR
            # x[i] = 2*x[Nxp+NG] - x[2*NG+2*Nxp-i]
            idx_b   = Nxp + NG
            idx_src = 2*NG + 2*Nxp - i
            x[i, j, k] = 2*x[idx_b, j, k] - x[idx_src, j, k]
            y[i, j, k] = 2*y[idx_b, j, k] - y[idx_src, j, k]
            z[i, j, k] = 2*z[idx_b, j, k] - z[idx_src, j, k]
        end
    end

    # ==========================================
    # 2. Y Direction Faces (Front & Back)
    # ==========================================

    # Front Boundary (Physical Y-)
    if ranky == 0
        @inbounds for k in K_in, i in I_in, j in J_gL
            idx_b   = NG + 1
            idx_src = 2*NG + 2 - j
            x[i, j, k] = 2*x[i, idx_b, k] - x[i, idx_src, k]
            y[i, j, k] = 2*y[i, idx_b, k] - y[i, idx_src, k]
            z[i, j, k] = 2*z[i, idx_b, k] - z[i, idx_src, k]
        end
    end

    # Back Boundary (Physical Y+)
    if ranky == Nprocs[2] - 1
        @inbounds for k in K_in, i in I_in, j in J_gR
            idx_b   = Nyp + NG
            idx_src = 2*NG + 2*Nyp - j
            x[i, j, k] = 2*x[i, idx_b, k] - x[i, idx_src, k]
            y[i, j, k] = 2*y[i, idx_b, k] - y[i, idx_src, k]
            z[i, j, k] = 2*z[i, idx_b, k] - z[i, idx_src, k]
        end
    end

    # ==========================================
    # 3. XY Corners (Ghost cells in both X and Y)
    # ==========================================
    
    # Corner: Left-Front (X- / Y-)
    if rankx == 0 && ranky == 0
        @inbounds for k in K_in, j in J_gL, i in I_gL
            # x[i,j] = x[i, NG+1] + x[NG+1, j] - x[NG+1, NG+1]
            x[i, j, k] = x[i, NG+1, k] + x[NG+1, j, k] - x[NG+1, NG+1, k]
            y[i, j, k] = y[i, NG+1, k] + y[NG+1, j, k] - y[NG+1, NG+1, k]
            z[i, j, k] = z[i, NG+1, k] + z[NG+1, j, k] - z[NG+1, NG+1, k]
        end
    end

    # Corner: Left-Back (X- / Y+)
    if rankx == 0 && ranky == Nprocs[2] - 1
        @inbounds for k in K_in, j in J_gR, i in I_gL
            # x[i,j] = x[i, Nyp+NG] + x[NG+1, j] - x[NG+1, Nyp+NG]
            x[i, j, k] = x[i, Nyp+NG, k] + x[NG+1, j, k] - x[NG+1, Nyp+NG, k]
            y[i, j, k] = y[i, Nyp+NG, k] + y[NG+1, j, k] - y[NG+1, Nyp+NG, k]
            z[i, j, k] = z[i, Nyp+NG, k] + z[NG+1, j, k] - z[NG+1, Nyp+NG, k]
        end
    end

    # Corner: Right-Front (X+ / Y-)
    if rankx == Nprocs[1] - 1 && ranky == 0
        @inbounds for k in K_in, j in J_gL, i in I_gR
            # x[i,j] = x[i, NG+1] + x[Nxp+NG, j] - x[Nxp+NG, NG+1]
            x[i, j, k] = x[i, NG+1, k] + x[Nxp+NG, j, k] - x[Nxp+NG, NG+1, k]
            y[i, j, k] = y[i, NG+1, k] + y[Nxp+NG, j, k] - y[Nxp+NG, NG+1, k]
            z[i, j, k] = z[i, NG+1, k] + z[Nxp+NG, j, k] - z[Nxp+NG, NG+1, k]
        end
    end

    # Corner: Right-Back (X+ / Y+)
    if rankx == Nprocs[1] - 1 && ranky == Nprocs[2] - 1
        @inbounds for k in K_in, j in J_gR, i in I_gR
            # x[i,j] = x[i, Nyp+NG] + x[Nxp+NG, j] - x[Nxp+NG, Nyp+NG]
            x[i, j, k] = x[i, Nyp+NG, k] + x[Nxp+NG, j, k] - x[Nxp+NG, Nyp+NG, k]
            y[i, j, k] = y[i, Nyp+NG, k] + y[Nxp+NG, j, k] - y[Nxp+NG, Nyp+NG, k]
            z[i, j, k] = z[i, Nyp+NG, k] + z[Nxp+NG, j, k] - z[Nxp+NG, Nyp+NG, k]
        end
    end

    # ==========================================
    # 4. Z Direction (Top & Bottom)
    # ==========================================
    # 注意：Z 方向使用 1:Nx_tot 和 1:Ny_tot，这意味着它会处理包括 X/Y Ghost 在内的所有区域
    # 因此 Z 方向的外推必须放在 X/Y 处理完之后
    
    local_Nx_tot = Nxp + 2*NG
    local_Ny_tot = Nyp + 2*NG

    # Bottom Boundary (Physical Z-)
    if rankz == 0
        @inbounds for k in K_gL, j in 1:local_Ny_tot, i in 1:local_Nx_tot
            idx_b   = NG + 1
            idx_src = 2*NG + 2 - k
            x[i, j, k] = 2*x[i, j, idx_b] - x[i, j, idx_src]
            y[i, j, k] = 2*y[i, j, idx_b] - y[i, j, idx_src]
            z[i, j, k] = 2*z[i, j, idx_b] - z[i, j, idx_src]
        end
    end

    # Top Boundary (Physical Z+)
    if rankz == Nprocs[3] - 1
        @inbounds for k in K_gR, j in 1:local_Ny_tot, i in 1:local_Nx_tot
            idx_b   = Nzp + NG
            idx_src = 2*NG + 2*Nzp - k
            x[i, j, k] = 2*x[i, j, idx_b] - x[i, j, idx_src]
            y[i, j, k] = 2*y[i, j, idx_b] - y[i, j, idx_src]
            z[i, j, k] = 2*z[i, j, idx_b] - z[i, j, idx_src]
        end
    end

    return
end