include("schemes.jl")

# const SCALE_FACTOR = 1.0f10
# const INV_SCALE_FACTOR = 1.0f-10

function Eigen_reconstruct_i(Q, U, ϕ, S, Fx, dξdx, dξdy, dξdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # 1. 边界检查
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG || j < NG+1 || k < NG+1
        return
    end
    # 2. 几何计算
    @inbounds nx = (dξdx[i, j, k] + dξdx[i+1, j, k]) * 0.5f0# * SCALE_FACTOR
    @inbounds ny = (dξdy[i, j, k] + dξdy[i+1, j, k]) * 0.5f0# * SCALE_FACTOR
    @inbounds nz = (dξdz[i, j, k] + dξdz[i+1, j, k]) * 0.5f0# * SCALE_FACTOR
    @inbounds Area = sqrt(nx*nx + ny*ny + nz*nz + 1.0f-20)
    @inbounds inv_len = 1.0f0 / Area
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len
    # @inbounds Area *= INV_SCALE_FACTOR
    # @cuprintf("nx=%e, ny=%e, nz=%e, Area=%e\n", nx, ny, nz, Area)

    # 3. 激波传感器
    @inbounds ϕx = max(ϕ[i-2, j, k], ϕ[i-1, j, k], ϕ[i, j, k], ϕ[i+1, j, k], ϕ[i+2, j, k], ϕ[i+3, j, k])

    # 准备最终累加变量
    UL_final_1 = 0.0f0; UL_final_2 = 0.0f0; UL_final_3 = 0.0f0; UL_final_4 = 0.0f0; UL_final_5 = 0.0f0
    UR_final_1 = 0.0f0; UR_final_2 = 0.0f0; UR_final_3 = 0.0f0; UR_final_4 = 0.0f0; UR_final_5 = 0.0f0

    # ==============================
    # 分支 A: 光滑区 (极速模式)
    # ==============================
    if ϕx < hybrid_ϕ1
        for n = 1:5
            @inbounds v1 = U[i-3,j,k,n]; v2 = U[i-2,j,k,n]; v3 = U[i-1,j,k,n]
            @inbounds v4 = U[i  ,j,k,n]; v5 = U[i+1,j,k,n]; v6 = U[i+2,j,k,n]; v7 = U[i+3,j,k,n]
            
            valL = Linear[1]*v1 + Linear[2]*v2 + Linear[3]*v3 + Linear[4]*v4 + Linear[5]*v5 + Linear[6]*v6 + Linear[7]*v7
            
            @inbounds r1 = U[i+4,j,k,n]; r2 = U[i+3,j,k,n]; r3 = U[i+2,j,k,n]
            @inbounds r4 = U[i+1,j,k,n]; r5 = U[i  ,j,k,n]; r6 = U[i-1,j,k,n]; r7 = U[i-2,j,k,n]
            
            valR = Linear[1]*r1 + Linear[2]*r2 + Linear[3]*r3 + Linear[4]*r4 + Linear[5]*r5 + Linear[6]*r6 + Linear[7]*r7

            if n==1; UL_final_1=valL; UR_final_1=valR
            elseif n==2; UL_final_2=valL; UR_final_2=valR
            elseif n==3; UL_final_3=valL; UR_final_3=valR
            elseif n==4; UL_final_4=valL; UR_final_4=valR
            else; UL_final_5=valL; UR_final_5=valR; end
        end

    # ==============================
    # 分支 B: 间断区 (特征分解模式)
    # ==============================
    else 
        @inbounds ρ = sqrt(Q[i, j, k, 1] * Q[i+1, j, k, 1])
        @inbounds u = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 2] + sqrt(Q[i+1, j, k, 1]) * Q[i+1, j, k, 2]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        @inbounds v = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 3] + sqrt(Q[i+1, j, k, 1]) * Q[i+1, j, k, 3]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        @inbounds w = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 4] + sqrt(Q[i+1, j, k, 1]) * Q[i+1, j, k, 4]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        @inbounds HL = γ/(γ-1.0f0)*Q[i, j, k, 5]/Q[i, j, k, 1] + 0.5f0*(Q[i, j, k, 2]^2 + Q[i, j, k, 3]^2 + Q[i, j, k, 4]^2)
        @inbounds HR = γ/(γ-1.0f0)*Q[i+1, j, k, 5]/Q[i+1, j, k, 1] + 0.5f0*(Q[i+1, j, k, 2]^2 + Q[i+1, j, k, 3]^2 + Q[i+1, j, k, 4]^2)
        @inbounds H = (sqrt(Q[i, j, k, 1]) * HL + sqrt(Q[i+1, j, k, 1]) * HR) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        
        v2 = 0.5f0*(u^2 + v^2 + w^2)
        c = sqrt((γ-1.0f0)*(H - v2))
        
        if abs(nz) <= abs(ny)
            den = sqrt(nx*nx + ny*ny + 1.0f-12); lx = -ny / den; ly = nx / den; lz = 0.0f0
        else
            den = sqrt(nx*nx + nz*nz + 1.0f-12); lx = -nz / den; ly = 0.0f0; lz = nx / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = 1.0f0/c; invc2 = invc*invc
        K = γ - 1.0f0
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = 0.5f0; mhalf = -0.5f0

        WENOϵ1 = 1.0f-10; WENOϵ2 = 1.0f-8
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0
        @inbounds ss = 2.0f0/(S[i+1, j, k] + S[i, j, k])

        for n = 1:5
            # 2a. 左特征向量 L
            ln1=0.0f0; ln2=0.0f0; ln3=0.0f0; ln4=0.0f0; ln5=0.0f0
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = 1.0f0 - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = 0.0f0
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = 0.0f0; end

            # 2b. 投影 U -> V (Local Characteristic Variables)
            @inbounds V1L = ln1*U[i-3,j,k,1] + ln2*U[i-3,j,k,2] + ln3*U[i-3,j,k,3] + ln4*U[i-3,j,k,4] + ln5*U[i-3,j,k,5]
            @inbounds V2L = ln1*U[i-2,j,k,1] + ln2*U[i-2,j,k,2] + ln3*U[i-2,j,k,3] + ln4*U[i-2,j,k,4] + ln5*U[i-2,j,k,5]
            @inbounds V3L = ln1*U[i-1,j,k,1] + ln2*U[i-1,j,k,2] + ln3*U[i-1,j,k,3] + ln4*U[i-1,j,k,4] + ln5*U[i-1,j,k,5]
            @inbounds V4L = ln1*U[i  ,j,k,1] + ln2*U[i  ,j,k,2] + ln3*U[i  ,j,k,3] + ln4*U[i  ,j,k,4] + ln5*U[i  ,j,k,5]
            @inbounds V5L = ln1*U[i+1,j,k,1] + ln2*U[i+1,j,k,2] + ln3*U[i+1,j,k,3] + ln4*U[i+1,j,k,4] + ln5*U[i+1,j,k,5]
            @inbounds V6L = ln1*U[i+2,j,k,1] + ln2*U[i+2,j,k,2] + ln3*U[i+2,j,k,3] + ln4*U[i+2,j,k,4] + ln5*U[i+2,j,k,5]
            @inbounds V7L = ln1*U[i+3,j,k,1] + ln2*U[i+3,j,k,2] + ln3*U[i+3,j,k,3] + ln4*U[i+3,j,k,4] + ln5*U[i+3,j,k,5]

            @inbounds V1R = ln1*U[i+4,j,k,1] + ln2*U[i+4,j,k,2] + ln3*U[i+4,j,k,3] + ln4*U[i+4,j,k,4] + ln5*U[i+4,j,k,5]
            @inbounds V2R = ln1*U[i+3,j,k,1] + ln2*U[i+3,j,k,2] + ln3*U[i+3,j,k,3] + ln4*U[i+3,j,k,4] + ln5*U[i+3,j,k,5]
            @inbounds V3R = ln1*U[i+2,j,k,1] + ln2*U[i+2,j,k,2] + ln3*U[i+2,j,k,3] + ln4*U[i+2,j,k,4] + ln5*U[i+2,j,k,5]
            @inbounds V4R = ln1*U[i+1,j,k,1] + ln2*U[i+1,j,k,2] + ln3*U[i+1,j,k,3] + ln4*U[i+1,j,k,4] + ln5*U[i+1,j,k,5]
            @inbounds V5R = ln1*U[i  ,j,k,1] + ln2*U[i  ,j,k,2] + ln3*U[i  ,j,k,3] + ln4*U[i  ,j,k,4] + ln5*U[i  ,j,k,5]
            @inbounds V6R = ln1*U[i-1,j,k,1] + ln2*U[i-1,j,k,2] + ln3*U[i-1,j,k,3] + ln4*U[i-1,j,k,4] + ln5*U[i-1,j,k,5]
            @inbounds V7R = ln1*U[i-2,j,k,1] + ln2*U[i-2,j,k,2] + ln3*U[i-2,j,k,3] + ln4*U[i-2,j,k,4] + ln5*U[i-2,j,k,5]

            valL = 0.0f0; valR = 0.0f0

            # 2c. WENO Reconstruction
            if ϕx < hybrid_ϕ2 # WENO7
                q1L = -3.0f0*V1L + 13.0f0*V2L - 23.0f0*V3L + 25.0f0*V4L; q1R = -3.0f0*V1R + 13.0f0*V2R - 23.0f0*V3R + 25.0f0*V4R
                q2L =  1.0f0*V2L -  5.0f0*V3L + 13.0f0*V4L +  3.0f0*V5L; q2R =  1.0f0*V2R -  5.0f0*V3R + 13.0f0*V4R +  3.0f0*V5R
                q3L = -1.0f0*V3L +  7.0f0*V4L +  7.0f0*V5L -  1.0f0*V6L; q3R = -1.0f0*V3R +  7.0f0*V4R +  7.0f0*V5R -  1.0f0*V6R
                q4L =  3.0f0*V4L + 13.0f0*V5L -  5.0f0*V6L +  1.0f0*V7L; q4R =  3.0f0*V4R + 13.0f0*V5R -  5.0f0*V6R +  1.0f0*V7R

                Is1L = V1L*( 547.0f0*V1L - 3882.0f0*V2L + 4642.0f0*V3L - 1854.0f0*V4L) + V2L*( 7043.0f0*V2L -17246.0f0*V3L + 7042.0f0*V4L) + V3L*(11003.0f0*V3L - 9402.0f0*V4L) + V4L*( 2107.0f0*V4L)
                Is2L = V2L*( 267.0f0*V2L - 1642.0f0*V3L + 1602.0f0*V4L -  494.0f0*V5L) + V3L*( 2843.0f0*V3L - 5966.0f0*V4L + 1922.0f0*V5L) + V4L*( 3443.0f0*V4L - 2522.0f0*V5L) + V5L*(  547.0f0*V5L)
                Is3L = V3L*( 547.0f0*V3L - 2522.0f0*V4L + 1922.0f0*V5L -  494.0f0*V6L) + V4L*( 3443.0f0*V4L - 5966.0f0*V5L + 1602.0f0*V6L) + V5L*( 2843.0f0*V5L - 1642.0f0*V6L) + V6L*(  267.0f0*V6L)
                Is4L = V4L*( 2107.0f0*V4L - 9402.0f0*V5L + 7042.0f0*V6L - 1854.0f0*V7L) + V5L*(11003.0f0*V5L -17246.0f0*V6L + 4642.0f0*V7L) + V6L*( 7043.0f0*V6L - 3882.0f0*V7L) + V7L*(  547.0f0*V7L)

                Is1R = V1R*( 547.0f0*V1R - 3882.0f0*V2R + 4642.0f0*V3R - 1854.0f0*V4R) + V2R*( 7043.0f0*V2R -17246.0f0*V3R + 7042.0f0*V4R) + V3R*(11003.0f0*V3R - 9402.0f0*V4R) + V4R*( 2107.0f0*V4R)
                Is2R = V2R*( 267.0f0*V2R - 1642.0f0*V3R + 1602.0f0*V4R -  494.0f0*V5R) + V3R*( 2843.0f0*V3R - 5966.0f0*V4R + 1922.0f0*V5R) + V4R*( 3443.0f0*V4R - 2522.0f0*V5R) + V5R*(  547.0f0*V5R)
                Is3R = V3R*( 547.0f0*V3R - 2522.0f0*V4R + 1922.0f0*V5R -  494.0f0*V6R) + V4R*( 3443.0f0*V4R - 5966.0f0*V5R + 1602.0f0*V6R) + V5R*( 2843.0f0*V5R - 1642.0f0*V6R) + V6R*(  267.0f0*V6R)
                Is4R = V4R*( 2107.0f0*V4R - 9402.0f0*V5R + 7042.0f0*V6R - 1854.0f0*V7R) + V5R*(11003.0f0*V5R -17246.0f0*V6R + 4642.0f0*V7R) + V6R*( 7043.0f0*V6R - 3882.0f0*V7R) + V7R*(  547.0f0*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(t_d1L * t_d1L);  t_d1R = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2L = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(t_d2L * t_d2L); t_d2R = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(t_d2R * t_d2R)
                t_d3L = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(t_d3L * t_d3L); t_d3R = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(t_d3R * t_d3R)
                t_d4L = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(t_d4L * t_d4L);  t_d4R = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(t_d4R * t_d4R)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1

                # ... (valL 和 valR 计算完毕) ...
            
            # ===  WENO5 分支 ===
            elseif ϕx < hybrid_ϕ3 # WENO5
                # 注意：WENO5 只需要 5 个点。
                # V2L (i-2), V3L (i-1), V4L (i), V5L (i+1), V6L (i+2)
                # 对应标准 WENO5 的 v1...v5
                
                # Left Side
                # Beta 1: (13/12)(v1-2v2+v3)^2 + (1/4)(v1-4v2+3v3)^2
                t1 = V2L - 2.0f0*V3L + V4L; t2 = V2L - 4.0f0*V3L + 3.0f0*V4L
                s1L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                
                # Beta 2: (13/12)(v2-2v3+v4)^2 + (1/4)(v2-v4)^2
                t1 = V3L - 2.0f0*V4L + V5L; t2 = V3L - V5L
                s2L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                
                # Beta 3: (13/12)(v3-2v4+v5)^2 + (1/4)(3v3-4v4+v5)^2
                t1 = V4L - 2.0f0*V5L + V6L; t2 = 3.0f0*V4L - 4.0f0*V5L + V6L
                s3L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                # Weights (d0=1/10, d1=6/10, d2=3/10 -> relative 1, 6, 3)
                t_d1L = WENOϵ2 + s1L*ss; α1L = 1.0f0/(t_d1L * t_d1L)
                t_d2L = WENOϵ2 + s2L*ss; α2L = 6.0f0/(t_d2L * t_d2L)
                t_d3L = WENOϵ2 + s3L*ss; α3L = 3.0f0/(t_d3L * t_d3L)
                invsumL = 1.0f0/(α1L+α2L+α3L)

                # Candidates
                v1 = 2.0f0*V2L - 7.0f0*V3L + 11.0f0*V4L
                v2 = -1.0f0*V3L + 5.0f0*V4L + 2.0f0*V5L
                v3 = 2.0f0*V4L + 5.0f0*V5L - 1.0f0*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2 # tmp2 is 1/6

                # Right Side (Symmetric)
                # Use V2R...V6R
                t1 = V2R - 2.0f0*V3R + V4R; t2 = V2R - 4.0f0*V3R + 3.0f0*V4R
                s1R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3R - 2.0f0*V4R + V5R; t2 = V3R - V5R
                s2R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4R - 2.0f0*V5R + V6R; t2 = 3.0f0*V4R - 4.0f0*V5R + V6R
                s3R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2R = WENOϵ2 + s2R*ss; α2R = 6.0f0/(t_d2R * t_d2R)
                t_d3R = WENOϵ2 + s3R*ss; α3R = 3.0f0/(t_d3R * t_d3R)
                invsumR = 1.0f0/(α1R+α2R+α3R)

                v1 = 2.0f0*V2R - 7.0f0*V3R + 11.0f0*V4R
                v2 = -1.0f0*V3R + 5.0f0*V4R + 2.0f0*V5R
                v3 = 2.0f0*V4R + 5.0f0*V5R - 1.0f0*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + 0.5f0*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - 0.5f0*minmod(V4R - V3R, V4R - V5R)
            end

            # 2d. 投影回物理空间
            rn1=0.0f0; rn2=0.0f0; rn3=0.0f0; rn4=0.0f0; rn5=0.0f0
            if n == 1; rn1=1.0f0; rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=1.0f0; rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=1.0f0; rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=0.0f0; rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=0.0f0; rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            UL_final_1 += rn1 * valL; UR_final_1 += rn1 * valR
            UL_final_2 += rn2 * valL; UR_final_2 += rn2 * valR
            UL_final_3 += rn3 * valL; UR_final_3 += rn3 * valR
            UL_final_4 += rn4 * valL; UR_final_4 += rn4 * valR
            UL_final_5 += rn5 * valL; UR_final_5 += rn5 * valR
        end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{5,Float32}(UL_final_1, UL_final_2, UL_final_3, UL_final_4, UL_final_5)
    UR_vec = SVector{5,Float32}(UR_final_1, UR_final_2, UR_final_3, UR_final_4, UR_final_5)
    
    flux_temp = HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)

    @inbounds begin
        Fx[i-NG+1, j-NG, k-NG, 1] = flux_temp[1] * Area
        Fx[i-NG+1, j-NG, k-NG, 2] = flux_temp[2] * Area
        Fx[i-NG+1, j-NG, k-NG, 3] = flux_temp[3] * Area
        Fx[i-NG+1, j-NG, k-NG, 4] = flux_temp[4] * Area
        Fx[i-NG+1, j-NG, k-NG, 5] = flux_temp[5] * Area
    end
    return 
end

function Eigen_reconstruct_j(Q, U, ϕ, S, Fy, dηdx, dηdy, dηdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # 1. 边界检查
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG || k < NG+1
        return
    end

    # 2. 几何计算 (使用 dη, j方向平均)
    @inbounds nx = (dηdx[i, j, k] + dηdx[i, j+1, k]) * 0.5f0
    @inbounds ny = (dηdy[i, j, k] + dηdy[i, j+1, k]) * 0.5f0
    @inbounds nz = (dηdz[i, j, k] + dηdz[i, j+1, k]) * 0.5f0
    @inbounds Area = sqrt(nx*nx + ny*ny + nz*nz + 1.0f-12)
    @inbounds inv_len = 1.0f0 / Area
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len

    # 3. 激波传感器 (J 方向)
    @inbounds ϕx = max(ϕ[i, j-2, k], ϕ[i, j-1, k], ϕ[i, j, k], ϕ[i, j+1, k], ϕ[i, j+2, k], ϕ[i, j+3, k])

    # 准备最终累加变量
    UL_final_1 = 0.0f0; UL_final_2 = 0.0f0; UL_final_3 = 0.0f0; UL_final_4 = 0.0f0; UL_final_5 = 0.0f0
    UR_final_1 = 0.0f0; UR_final_2 = 0.0f0; UR_final_3 = 0.0f0; UR_final_4 = 0.0f0; UR_final_5 = 0.0f0

    # ==============================
    # 分支 A: 光滑区 (极速模式)
    # ==============================
    if ϕx < hybrid_ϕ1
        for n = 1:5
            # Load Stencil along J
            @inbounds v1 = U[i,j-3,k,n]; v2 = U[i,j-2,k,n]; v3 = U[i,j-1,k,n]
            @inbounds v4 = U[i,j  ,k,n]; v5 = U[i,j+1,k,n]; v6 = U[i,j+2,k,n]; v7 = U[i,j+3,k,n]
            
            valL = Linear[1]*v1 + Linear[2]*v2 + Linear[3]*v3 + Linear[4]*v4 + Linear[5]*v5 + Linear[6]*v6 + Linear[7]*v7
            
            # Right State (Mirror in J)
            @inbounds r1 = U[i,j+4,k,n]; r2 = U[i,j+3,k,n]; r3 = U[i,j+2,k,n]
            @inbounds r4 = U[i,j+1,k,n]; r5 = U[i,j  ,k,n]; r6 = U[i,j-1,k,n]; r7 = U[i,j-2,k,n]
            
            valR = Linear[1]*r1 + Linear[2]*r2 + Linear[3]*r3 + Linear[4]*r4 + Linear[5]*r5 + Linear[6]*r6 + Linear[7]*r7

            if n==1; UL_final_1=valL; UR_final_1=valR
            elseif n==2; UL_final_2=valL; UR_final_2=valR
            elseif n==3; UL_final_3=valL; UR_final_3=valR
            elseif n==4; UL_final_4=valL; UR_final_4=valR
            else; UL_final_5=valL; UR_final_5=valR; end
        end

    # ==============================
    # 分支 B: 间断区
    # ==============================
    else 
        # Roe Averages (J direction: j and j+1)
        @inbounds ρ = sqrt(Q[i, j, k, 1] * Q[i, j+1, k, 1])
        @inbounds u = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 2] + sqrt(Q[i, j+1, k, 1]) * Q[i, j+1, k, 2]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        @inbounds v = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 3] + sqrt(Q[i, j+1, k, 1]) * Q[i, j+1, k, 3]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        @inbounds w = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 4] + sqrt(Q[i, j+1, k, 1]) * Q[i, j+1, k, 4]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        @inbounds HL = γ/(γ-1.0f0)*Q[i, j, k, 5]/Q[i, j, k, 1] + 0.5f0*(Q[i, j, k, 2]^2 + Q[i, j, k, 3]^2 + Q[i, j, k, 4]^2)
        @inbounds HR = γ/(γ-1.0f0)*Q[i, j+1, k, 5]/Q[i, j+1, k, 1] + 0.5f0*(Q[i, j+1, k, 2]^2 + Q[i, j+1, k, 3]^2 + Q[i, j+1, k, 4]^2)
        @inbounds H = (sqrt(Q[i, j, k, 1]) * HL + sqrt(Q[i, j+1, k, 1]) * HR) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        
        v2 = 0.5f0*(u^2 + v^2 + w^2)
        c = sqrt((γ-1.0f0)*(H - v2))
        
        # Tangent vectors
        if abs(nz) <= abs(ny)
            den = sqrt(nx*nx + ny*ny + 1.0f-12); lx = -ny / den; ly = nx / den; lz = 0.0f0
        else
            den = sqrt(nx*nx + nz*nz + 1.0f-12); lx = -nz / den; ly = 0.0f0; lz = nx / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = 1.0f0/c; invc2 = invc*invc
        K = γ - 1.0f0
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = 0.5f0; mhalf = -0.5f0

        WENOϵ1 = 1.0f-10; WENOϵ2 = 1.0f-8
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0
        @inbounds ss = 2.0f0/(S[i, j+1, k] + S[i, j, k]) # Note: S indices changed

        for n = 1:5
            # 2a. 左特征向量 L
            ln1=0.0f0; ln2=0.0f0; ln3=0.0f0; ln4=0.0f0; ln5=0.0f0
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = 1.0f0 - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = 0.0f0
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = 0.0f0; end

            # 2b. 投影 U -> V (Stencil along J)
            @inbounds V1L = ln1*U[i,j-3,k,1] + ln2*U[i,j-3,k,2] + ln3*U[i,j-3,k,3] + ln4*U[i,j-3,k,4] + ln5*U[i,j-3,k,5]
            @inbounds V2L = ln1*U[i,j-2,k,1] + ln2*U[i,j-2,k,2] + ln3*U[i,j-2,k,3] + ln4*U[i,j-2,k,4] + ln5*U[i,j-2,k,5]
            @inbounds V3L = ln1*U[i,j-1,k,1] + ln2*U[i,j-1,k,2] + ln3*U[i,j-1,k,3] + ln4*U[i,j-1,k,4] + ln5*U[i,j-1,k,5]
            @inbounds V4L = ln1*U[i,j  ,k,1] + ln2*U[i,j  ,k,2] + ln3*U[i,j  ,k,3] + ln4*U[i,j  ,k,4] + ln5*U[i,j  ,k,5]
            @inbounds V5L = ln1*U[i,j+1,k,1] + ln2*U[i,j+1,k,2] + ln3*U[i,j+1,k,3] + ln4*U[i,j+1,k,4] + ln5*U[i,j+1,k,5]
            @inbounds V6L = ln1*U[i,j+2,k,1] + ln2*U[i,j+2,k,2] + ln3*U[i,j+2,k,3] + ln4*U[i,j+2,k,4] + ln5*U[i,j+2,k,5]
            @inbounds V7L = ln1*U[i,j+3,k,1] + ln2*U[i,j+3,k,2] + ln3*U[i,j+3,k,3] + ln4*U[i,j+3,k,4] + ln5*U[i,j+3,k,5]

            @inbounds V1R = ln1*U[i,j+4,k,1] + ln2*U[i,j+4,k,2] + ln3*U[i,j+4,k,3] + ln4*U[i,j+4,k,4] + ln5*U[i,j+4,k,5]
            @inbounds V2R = ln1*U[i,j+3,k,1] + ln2*U[i,j+3,k,2] + ln3*U[i,j+3,k,3] + ln4*U[i,j+3,k,4] + ln5*U[i,j+3,k,5]
            @inbounds V3R = ln1*U[i,j+2,k,1] + ln2*U[i,j+2,k,2] + ln3*U[i,j+2,k,3] + ln4*U[i,j+2,k,4] + ln5*U[i,j+2,k,5]
            @inbounds V4R = ln1*U[i,j+1,k,1] + ln2*U[i,j+1,k,2] + ln3*U[i,j+1,k,3] + ln4*U[i,j+1,k,4] + ln5*U[i,j+1,k,5]
            @inbounds V5R = ln1*U[i,j  ,k,1] + ln2*U[i,j  ,k,2] + ln3*U[i,j  ,k,3] + ln4*U[i,j  ,k,4] + ln5*U[i,j  ,k,5]
            @inbounds V6R = ln1*U[i,j-1,k,1] + ln2*U[i,j-1,k,2] + ln3*U[i,j-1,k,3] + ln4*U[i,j-1,k,4] + ln5*U[i,j-1,k,5]
            @inbounds V7R = ln1*U[i,j-2,k,1] + ln2*U[i,j-2,k,2] + ln3*U[i,j-2,k,3] + ln4*U[i,j-2,k,4] + ln5*U[i,j-2,k,5]

            valL = 0.0f0; valR = 0.0f0

            # 2c. WENO Reconstruction (Pure math, identical to i-dir)
            if ϕx < hybrid_ϕ2 # WENO7
                q1L = -3.0f0*V1L + 13.0f0*V2L - 23.0f0*V3L + 25.0f0*V4L; q1R = -3.0f0*V1R + 13.0f0*V2R - 23.0f0*V3R + 25.0f0*V4R
                q2L =  1.0f0*V2L -  5.0f0*V3L + 13.0f0*V4L +  3.0f0*V5L; q2R =  1.0f0*V2R -  5.0f0*V3R + 13.0f0*V4R +  3.0f0*V5R
                q3L = -1.0f0*V3L +  7.0f0*V4L +  7.0f0*V5L -  1.0f0*V6L; q3R = -1.0f0*V3R +  7.0f0*V4R +  7.0f0*V5R -  1.0f0*V6R
                q4L =  3.0f0*V4L + 13.0f0*V5L -  5.0f0*V6L +  1.0f0*V7L; q4R =  3.0f0*V4R + 13.0f0*V5R -  5.0f0*V6R +  1.0f0*V7R

                Is1L = V1L*( 547.0f0*V1L - 3882.0f0*V2L + 4642.0f0*V3L - 1854.0f0*V4L) + V2L*( 7043.0f0*V2L -17246.0f0*V3L + 7042.0f0*V4L) + V3L*(11003.0f0*V3L - 9402.0f0*V4L) + V4L*( 2107.0f0*V4L)
                Is2L = V2L*( 267.0f0*V2L - 1642.0f0*V3L + 1602.0f0*V4L -  494.0f0*V5L) + V3L*( 2843.0f0*V3L - 5966.0f0*V4L + 1922.0f0*V5L) + V4L*( 3443.0f0*V4L - 2522.0f0*V5L) + V5L*(  547.0f0*V5L)
                Is3L = V3L*( 547.0f0*V3L - 2522.0f0*V4L + 1922.0f0*V5L -  494.0f0*V6L) + V4L*( 3443.0f0*V4L - 5966.0f0*V5L + 1602.0f0*V6L) + V5L*( 2843.0f0*V5L - 1642.0f0*V6L) + V6L*(  267.0f0*V6L)
                Is4L = V4L*( 2107.0f0*V4L - 9402.0f0*V5L + 7042.0f0*V6L - 1854.0f0*V7L) + V5L*(11003.0f0*V5L -17246.0f0*V6L + 4642.0f0*V7L) + V6L*( 7043.0f0*V6L - 3882.0f0*V7L) + V7L*(  547.0f0*V7L)

                Is1R = V1R*( 547.0f0*V1R - 3882.0f0*V2R + 4642.0f0*V3R - 1854.0f0*V4R) + V2R*( 7043.0f0*V2R -17246.0f0*V3R + 7042.0f0*V4R) + V3R*(11003.0f0*V3R - 9402.0f0*V4R) + V4R*( 2107.0f0*V4R)
                Is2R = V2R*( 267.0f0*V2R - 1642.0f0*V3R + 1602.0f0*V4R -  494.0f0*V5R) + V3R*( 2843.0f0*V3R - 5966.0f0*V4R + 1922.0f0*V5R) + V4R*( 3443.0f0*V4R - 2522.0f0*V5R) + V5R*(  547.0f0*V5R)
                Is3R = V3R*( 547.0f0*V3R - 2522.0f0*V4R + 1922.0f0*V5R -  494.0f0*V6R) + V4R*( 3443.0f0*V4R - 5966.0f0*V5R + 1602.0f0*V6R) + V5R*( 2843.0f0*V5R - 1642.0f0*V6R) + V6R*(  267.0f0*V6R)
                Is4R = V4R*( 2107.0f0*V4R - 9402.0f0*V5R + 7042.0f0*V6R - 1854.0f0*V7R) + V5R*(11003.0f0*V5R -17246.0f0*V6R + 4642.0f0*V7R) + V6R*( 7043.0f0*V6R - 3882.0f0*V7R) + V7R*(  547.0f0*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(t_d1L * t_d1L); t_d1R = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2L = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(t_d2L * t_d2L); t_d2R = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(t_d2R * t_d2R)
                t_d3L = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(t_d3L * t_d3L); t_d3R = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(t_d3R * t_d3R)
                t_d4L = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(t_d4L * t_d4L); t_d4R = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(t_d4R * t_d4R)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1
            
            elseif ϕx < hybrid_ϕ3 # WENO5
                # Left Side
                t1 = V2L - 2.0f0*V3L + V4L; t2 = V2L - 4.0f0*V3L + 3.0f0*V4L; s1L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3L - 2.0f0*V4L + V5L; t2 = V3L - V5L; s2L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4L - 2.0f0*V5L + V6L; t2 = 3.0f0*V4L - 4.0f0*V5L + V6L; s3L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                
                t_d1L = WENOϵ2 + s1L*ss; α1L = 1.0f0/(t_d1L * t_d1L)
                t_d2L = WENOϵ2 + s2L*ss; α2L = 6.0f0/(t_d2L * t_d2L)
                t_d3L = WENOϵ2 + s3L*ss; α3L = 3.0f0/(t_d3L * t_d3L)
                invsumL = 1.0f0/(α1L+α2L+α3L)

                v1 = 2.0f0*V2L - 7.0f0*V3L + 11.0f0*V4L
                v2 = -1.0f0*V3L + 5.0f0*V4L + 2.0f0*V5L
                v3 = 2.0f0*V4L + 5.0f0*V5L - 1.0f0*V6L
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2 

                # Right Side
                t1 = V2R - 2.0f0*V3R + V4R; t2 = V2R - 4.0f0*V3R + 3.0f0*V4R; s1R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3R - 2.0f0*V4R + V5R; t2 = V3R - V5R; s2R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4R - 2.0f0*V5R + V6R; t2 = 3.0f0*V4R - 4.0f0*V5R + V6R; s3R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2R = WENOϵ2 + s2R*ss; α2R = 6.0f0/(t_d2R * t_d2R)
                t_d3R = WENOϵ2 + s3R*ss; α3R = 3.0f0/(t_d3R * t_d3R)
                invsumR = 1.0f0/(α1R+α2R+α3R)

                v1 = 2.0f0*V2R - 7.0f0*V3R + 11.0f0*V4R
                v2 = -1.0f0*V3R + 5.0f0*V4R + 2.0f0*V5R
                v3 = 2.0f0*V4R + 5.0f0*V5R - 1.0f0*V6R
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + 0.5f0*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - 0.5f0*minmod(V4R - V3R, V4R - V5R)
            end

            # 2d. 投影回物理空间
            rn1=0.0f0; rn2=0.0f0; rn3=0.0f0; rn4=0.0f0; rn5=0.0f0
            if n == 1; rn1=1.0f0; rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=1.0f0; rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=1.0f0; rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=0.0f0; rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=0.0f0; rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            UL_final_1 += rn1 * valL; UR_final_1 += rn1 * valR
            UL_final_2 += rn2 * valL; UR_final_2 += rn2 * valR
            UL_final_3 += rn3 * valL; UR_final_3 += rn3 * valR
            UL_final_4 += rn4 * valL; UR_final_4 += rn4 * valR
            UL_final_5 += rn5 * valL; UR_final_5 += rn5 * valR
        end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{5,Float32}(UL_final_1, UL_final_2, UL_final_3, UL_final_4, UL_final_5)
    UR_vec = SVector{5,Float32}(UR_final_1, UR_final_2, UR_final_3, UR_final_4, UR_final_5)
    
    flux_temp = HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)

    @inbounds begin
        Fy[i-NG, j-NG+1, k-NG, 1] = flux_temp[1] * Area
        Fy[i-NG, j-NG+1, k-NG, 2] = flux_temp[2] * Area
        Fy[i-NG, j-NG+1, k-NG, 3] = flux_temp[3] * Area
        Fy[i-NG, j-NG+1, k-NG, 4] = flux_temp[4] * Area
        Fy[i-NG, j-NG+1, k-NG, 5] = flux_temp[5] * Area
    end
    return
end

function Eigen_reconstruct_k(Q, U, ϕ, S, Fz, dζdx, dζdy, dζdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # k 需要 stencil k-3 和 k+4
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG
        return
    end

    # 2. 几何计算 (使用 dζ, k方向平均)
    @inbounds nx = (dζdx[i, j, k] + dζdx[i, j, k+1]) * 0.5f0
    @inbounds ny = (dζdy[i, j, k] + dζdy[i, j, k+1]) * 0.5f0
    @inbounds nz = (dζdz[i, j, k] + dζdz[i, j, k+1]) * 0.5f0
    @inbounds Area = sqrt(nx*nx + ny*ny + nz*nz + 1.0f-12)
    @inbounds inv_len = 1.0f0 / Area
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len

    # 3. 激波传感器 (K 方向)
    @inbounds ϕx = max(ϕ[i, j, k-2], ϕ[i, j, k-1], ϕ[i, j, k], ϕ[i, j, k+1], ϕ[i, j, k+2], ϕ[i, j, k+3])

    # 准备最终累加变量
    UL_final_1 = 0.0f0; UL_final_2 = 0.0f0; UL_final_3 = 0.0f0; UL_final_4 = 0.0f0; UL_final_5 = 0.0f0
    UR_final_1 = 0.0f0; UR_final_2 = 0.0f0; UR_final_3 = 0.0f0; UR_final_4 = 0.0f0; UR_final_5 = 0.0f0

    # ==============================
    # 分支 A: 光滑区
    # ==============================
    if ϕx < hybrid_ϕ1
        for n = 1:5
            # Load Stencil along K
            @inbounds v1 = U[i,j,k-3,n]; v2 = U[i,j,k-2,n]; v3 = U[i,j,k-1,n]
            @inbounds v4 = U[i,j,k  ,n]; v5 = U[i,j,k+1,n]; v6 = U[i,j,k+2,n]; v7 = U[i,j,k+3,n]
            
            valL = Linear[1]*v1 + Linear[2]*v2 + Linear[3]*v3 + Linear[4]*v4 + Linear[5]*v5 + Linear[6]*v6 + Linear[7]*v7
            
            # Right State (Mirror in K)
            @inbounds r1 = U[i,j,k+4,n]; r2 = U[i,j,k+3,n]; r3 = U[i,j,k+2,n]
            @inbounds r4 = U[i,j,k+1,n]; r5 = U[i,j,k  ,n]; r6 = U[i,j,k-1,n]; r7 = U[i,j,k-2,n]
            
            valR = Linear[1]*r1 + Linear[2]*r2 + Linear[3]*r3 + Linear[4]*r4 + Linear[5]*r5 + Linear[6]*r6 + Linear[7]*r7

            if n==1; UL_final_1=valL; UR_final_1=valR
            elseif n==2; UL_final_2=valL; UR_final_2=valR
            elseif n==3; UL_final_3=valL; UR_final_3=valR
            elseif n==4; UL_final_4=valL; UR_final_4=valR
            else; UL_final_5=valL; UR_final_5=valR; end
        end

    # ==============================
    # 分支 B: 间断区
    # ==============================
    else 
        # Roe Averages (K direction: k and k+1)
        @inbounds ρ = sqrt(Q[i, j, k, 1] * Q[i, j, k+1, 1])
        @inbounds u = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 2] + sqrt(Q[i, j, k+1, 1]) * Q[i, j, k+1, 2]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        @inbounds v = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 3] + sqrt(Q[i, j, k+1, 1]) * Q[i, j, k+1, 3]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        @inbounds w = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 4] + sqrt(Q[i, j, k+1, 1]) * Q[i, j, k+1, 4]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        @inbounds HL = γ/(γ-1.0f0)*Q[i, j, k, 5]/Q[i, j, k, 1] + 0.5f0*(Q[i, j, k, 2]^2 + Q[i, j, k, 3]^2 + Q[i, j, k, 4]^2)
        @inbounds HR = γ/(γ-1.0f0)*Q[i, j, k+1, 5]/Q[i, j, k+1, 1] + 0.5f0*(Q[i, j, k+1, 2]^2 + Q[i, j, k+1, 3]^2 + Q[i, j, k+1, 4]^2)
        @inbounds H = (sqrt(Q[i, j, k, 1]) * HL + sqrt(Q[i, j, k+1, 1]) * HR) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        
        v2 = 0.5f0*(u^2 + v^2 + w^2)
        c = sqrt((γ-1.0f0)*(H - v2))
        
        # Tangent vectors
        if abs(nz) <= abs(ny)
            den = sqrt(nx*nx + ny*ny + 1.0f-12); lx = -ny / den; ly = nx / den; lz = 0.0f0
        else
            den = sqrt(nx*nx + nz*nz + 1.0f-12); lx = -nz / den; ly = 0.0f0; lz = nx / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = 1.0f0/c; invc2 = invc*invc
        K = γ - 1.0f0
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = 0.5f0; mhalf = -0.5f0

        WENOϵ1 = 1.0f-10; WENOϵ2 = 1.0f-8
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0
        @inbounds ss = 2.0f0/(S[i, j, k+1] + S[i, j, k]) # Note: S indices changed

        for n = 1:5
            # 2a. 左特征向量 L
            ln1=0.0f0; ln2=0.0f0; ln3=0.0f0; ln4=0.0f0; ln5=0.0f0
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = 1.0f0 - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = 0.0f0
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = 0.0f0; end

            # 2b. 投影 U -> V (Stencil along K)
            @inbounds V1L = ln1*U[i,j,k-3,1] + ln2*U[i,j,k-3,2] + ln3*U[i,j,k-3,3] + ln4*U[i,j,k-3,4] + ln5*U[i,j,k-3,5]
            @inbounds V2L = ln1*U[i,j,k-2,1] + ln2*U[i,j,k-2,2] + ln3*U[i,j,k-2,3] + ln4*U[i,j,k-2,4] + ln5*U[i,j,k-2,5]
            @inbounds V3L = ln1*U[i,j,k-1,1] + ln2*U[i,j,k-1,2] + ln3*U[i,j,k-1,3] + ln4*U[i,j,k-1,4] + ln5*U[i,j,k-1,5]
            @inbounds V4L = ln1*U[i,j,k  ,1] + ln2*U[i,j,k  ,2] + ln3*U[i,j,k  ,3] + ln4*U[i,j,k  ,4] + ln5*U[i,j,k  ,5]
            @inbounds V5L = ln1*U[i,j,k+1,1] + ln2*U[i,j,k+1,2] + ln3*U[i,j,k+1,3] + ln4*U[i,j,k+1,4] + ln5*U[i,j,k+1,5]
            @inbounds V6L = ln1*U[i,j,k+2,1] + ln2*U[i,j,k+2,2] + ln3*U[i,j,k+2,3] + ln4*U[i,j,k+2,4] + ln5*U[i,j,k+2,5]
            @inbounds V7L = ln1*U[i,j,k+3,1] + ln2*U[i,j,k+3,2] + ln3*U[i,j,k+3,3] + ln4*U[i,j,k+3,4] + ln5*U[i,j,k+3,5]

            @inbounds V1R = ln1*U[i,j,k+4,1] + ln2*U[i,j,k+4,2] + ln3*U[i,j,k+4,3] + ln4*U[i,j,k+4,4] + ln5*U[i,j,k+4,5]
            @inbounds V2R = ln1*U[i,j,k+3,1] + ln2*U[i,j,k+3,2] + ln3*U[i,j,k+3,3] + ln4*U[i,j,k+3,4] + ln5*U[i,j,k+3,5]
            @inbounds V3R = ln1*U[i,j,k+2,1] + ln2*U[i,j,k+2,2] + ln3*U[i,j,k+2,3] + ln4*U[i,j,k+2,4] + ln5*U[i,j,k+2,5]
            @inbounds V4R = ln1*U[i,j,k+1,1] + ln2*U[i,j,k+1,2] + ln3*U[i,j,k+1,3] + ln4*U[i,j,k+1,4] + ln5*U[i,j,k+1,5]
            @inbounds V5R = ln1*U[i,j,k  ,1] + ln2*U[i,j,k  ,2] + ln3*U[i,j,k  ,3] + ln4*U[i,j,k  ,4] + ln5*U[i,j,k  ,5]
            @inbounds V6R = ln1*U[i,j,k-1,1] + ln2*U[i,j,k-1,2] + ln3*U[i,j,k-1,3] + ln4*U[i,j,k-1,4] + ln5*U[i,j,k-1,5]
            @inbounds V7R = ln1*U[i,j,k-2,1] + ln2*U[i,j,k-2,2] + ln3*U[i,j,k-2,3] + ln4*U[i,j,k-2,4] + ln5*U[i,j,k-2,5]

            valL = 0.0f0; valR = 0.0f0

            # 2c. WENO Reconstruction (Pure math, identical to i-dir)
            if ϕx < hybrid_ϕ2 # WENO7
                q1L = -3.0f0*V1L + 13.0f0*V2L - 23.0f0*V3L + 25.0f0*V4L; q1R = -3.0f0*V1R + 13.0f0*V2R - 23.0f0*V3R + 25.0f0*V4R
                q2L =  1.0f0*V2L -  5.0f0*V3L + 13.0f0*V4L +  3.0f0*V5L; q2R =  1.0f0*V2R -  5.0f0*V3R + 13.0f0*V4R +  3.0f0*V5R
                q3L = -1.0f0*V3L +  7.0f0*V4L +  7.0f0*V5L -  1.0f0*V6L; q3R = -1.0f0*V3R +  7.0f0*V4R +  7.0f0*V5R -  1.0f0*V6R
                q4L =  3.0f0*V4L + 13.0f0*V5L -  5.0f0*V6L +  1.0f0*V7L; q4R =  3.0f0*V4R + 13.0f0*V5R -  5.0f0*V6R +  1.0f0*V7R

                Is1L = V1L*( 547.0f0*V1L - 3882.0f0*V2L + 4642.0f0*V3L - 1854.0f0*V4L) + V2L*( 7043.0f0*V2L -17246.0f0*V3L + 7042.0f0*V4L) + V3L*(11003.0f0*V3L - 9402.0f0*V4L) + V4L*( 2107.0f0*V4L)
                Is2L = V2L*( 267.0f0*V2L - 1642.0f0*V3L + 1602.0f0*V4L -  494.0f0*V5L) + V3L*( 2843.0f0*V3L - 5966.0f0*V4L + 1922.0f0*V5L) + V4L*( 3443.0f0*V4L - 2522.0f0*V5L) + V5L*(  547.0f0*V5L)
                Is3L = V3L*( 547.0f0*V3L - 2522.0f0*V4L + 1922.0f0*V5L -  494.0f0*V6L) + V4L*( 3443.0f0*V4L - 5966.0f0*V5L + 1602.0f0*V6L) + V5L*( 2843.0f0*V5L - 1642.0f0*V6L) + V6L*(  267.0f0*V6L)
                Is4L = V4L*( 2107.0f0*V4L - 9402.0f0*V5L + 7042.0f0*V6L - 1854.0f0*V7L) + V5L*(11003.0f0*V5L -17246.0f0*V6L + 4642.0f0*V7L) + V6L*( 7043.0f0*V6L - 3882.0f0*V7L) + V7L*(  547.0f0*V7L)

                Is1R = V1R*( 547.0f0*V1R - 3882.0f0*V2R + 4642.0f0*V3R - 1854.0f0*V4R) + V2R*( 7043.0f0*V2R -17246.0f0*V3R + 7042.0f0*V4R) + V3R*(11003.0f0*V3R - 9402.0f0*V4R) + V4R*( 2107.0f0*V4R)
                Is2R = V2R*( 267.0f0*V2R - 1642.0f0*V3R + 1602.0f0*V4R -  494.0f0*V5R) + V3R*( 2843.0f0*V3R - 5966.0f0*V4R + 1922.0f0*V5R) + V4R*( 3443.0f0*V4R - 2522.0f0*V5R) + V5R*(  547.0f0*V5R)
                Is3R = V3R*( 547.0f0*V3R - 2522.0f0*V4R + 1922.0f0*V5R -  494.0f0*V6R) + V4R*( 3443.0f0*V4R - 5966.0f0*V5R + 1602.0f0*V6R) + V5R*( 2843.0f0*V5R - 1642.0f0*V6R) + V6R*(  267.0f0*V6R)
                Is4R = V4R*( 2107.0f0*V4R - 9402.0f0*V5R + 7042.0f0*V6R - 1854.0f0*V7R) + V5R*(11003.0f0*V5R -17246.0f0*V6R + 4642.0f0*V7R) + V6R*( 7043.0f0*V6R - 3882.0f0*V7R) + V7R*(  547.0f0*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(t_d1L * t_d1L); t_d1R = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2L = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(t_d2L * t_d2L); t_d2R = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(t_d2R * t_d2R)
                t_d3L = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(t_d3L * t_d3L); t_d3R = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(t_d3R * t_d3R)
                t_d4L = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(t_d4L * t_d4L); t_d4R = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(t_d4R * t_d4R)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1
            
            elseif ϕx < hybrid_ϕ3 # WENO5
                # Left Side
                t1 = V2L - 2.0f0*V3L + V4L; t2 = V2L - 4.0f0*V3L + 3.0f0*V4L; s1L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3L - 2.0f0*V4L + V5L; t2 = V3L - V5L; s2L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4L - 2.0f0*V5L + V6L; t2 = 3.0f0*V4L - 4.0f0*V5L + V6L; s3L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                
                t_d1L = WENOϵ2 + s1L*ss; α1L = 1.0f0/(t_d1L * t_d1L)
                t_d2L = WENOϵ2 + s2L*ss; α2L = 6.0f0/(t_d2L * t_d2L)
                t_d3L = WENOϵ2 + s3L*ss; α3L = 3.0f0/(t_d3L * t_d3L)
                invsumL = 1.0f0/(α1L+α2L+α3L)

                v1 = 2.0f0*V2L - 7.0f0*V3L + 11.0f0*V4L
                v2 = -1.0f0*V3L + 5.0f0*V4L + 2.0f0*V5L
                v3 = 2.0f0*V4L + 5.0f0*V5L - 1.0f0*V6L
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2 

                # Right Side
                t1 = V2R - 2.0f0*V3R + V4R; t2 = V2R - 4.0f0*V3R + 3.0f0*V4R; s1R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3R - 2.0f0*V4R + V5R; t2 = V3R - V5R; s2R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4R - 2.0f0*V5R + V6R; t2 = 3.0f0*V4R - 4.0f0*V5R + V6R; s3R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2R = WENOϵ2 + s2R*ss; α2R = 6.0f0/(t_d2R * t_d2R)
                t_d3R = WENOϵ2 + s3R*ss; α3R = 3.0f0/(t_d3R * t_d3R)
                invsumR = 1.0f0/(α1R+α2R+α3R)

                v1 = 2.0f0*V2R - 7.0f0*V3R + 11.0f0*V4R
                v2 = -1.0f0*V3R + 5.0f0*V4R + 2.0f0*V5R
                v3 = 2.0f0*V4R + 5.0f0*V5R - 1.0f0*V6R
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + 0.5f0*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - 0.5f0*minmod(V4R - V3R, V4R - V5R)
            end

            # 2d. 投影回物理空间
            rn1=0.0f0; rn2=0.0f0; rn3=0.0f0; rn4=0.0f0; rn5=0.0f0
            if n == 1; rn1=1.0f0; rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=1.0f0; rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=1.0f0; rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=0.0f0; rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=0.0f0; rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            UL_final_1 += rn1 * valL; UR_final_1 += rn1 * valR
            UL_final_2 += rn2 * valL; UR_final_2 += rn2 * valR
            UL_final_3 += rn3 * valL; UR_final_3 += rn3 * valR
            UL_final_4 += rn4 * valL; UR_final_4 += rn4 * valR
            UL_final_5 += rn5 * valL; UR_final_5 += rn5 * valR
        end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{5,Float32}(UL_final_1, UL_final_2, UL_final_3, UL_final_4, UL_final_5)
    UR_vec = SVector{5,Float32}(UR_final_1, UR_final_2, UR_final_3, UR_final_4, UR_final_5)
    
    flux_temp = HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)

    @inbounds begin
        # 注意：写入索引 Fz 的维度是 (Nxp, Nyp, Nzp+1, 5)
        Fz[i-NG, j-NG, k-NG+1, 1] = flux_temp[1] * Area
        Fz[i-NG, j-NG, k-NG+1, 2] = flux_temp[2] * Area
        Fz[i-NG, j-NG, k-NG+1, 3] = flux_temp[3] * Area
        Fz[i-NG, j-NG, k-NG+1, 4] = flux_temp[4] * Area
        Fz[i-NG, j-NG, k-NG+1, 5] = flux_temp[5] * Area
    end
    return
end

function Conser_reconstruct_i(Q, U, ϕ, S, Fx, dξdx, dξdy, dξdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # 1. 边界检查
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG || j < NG+1 || k < NG+1
        return
    end
    # 2. 几何计算
    @inbounds nx = (dξdx[i, j, k] + dξdx[i+1, j, k]) * 0.5f0# * SCALE_FACTOR
    @inbounds ny = (dξdy[i, j, k] + dξdy[i+1, j, k]) * 0.5f0# * SCALE_FACTOR
    @inbounds nz = (dξdz[i, j, k] + dξdz[i+1, j, k]) * 0.5f0# * SCALE_FACTOR
    @inbounds Area = sqrt(nx*nx + ny*ny + nz*nz + 1.0f-20)
    @inbounds inv_len = 1.0f0 / Area
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len
    # @inbounds Area *= INV_SCALE_FACTOR
    # @cuprintf("nx=%e, ny=%e, nz=%e, Area=%e\n", nx, ny, nz, Area)

    # 3. 激波传感器
    @inbounds ϕx = max(ϕ[i-2, j, k], ϕ[i-1, j, k], ϕ[i, j, k], ϕ[i+1, j, k], ϕ[i+2, j, k], ϕ[i+3, j, k])

    # 准备最终累加变量
    UL_final_1 = 0.0f0; UL_final_2 = 0.0f0; UL_final_3 = 0.0f0; UL_final_4 = 0.0f0; UL_final_5 = 0.0f0
    UR_final_1 = 0.0f0; UR_final_2 = 0.0f0; UR_final_3 = 0.0f0; UR_final_4 = 0.0f0; UR_final_5 = 0.0f0

    # ==============================
    # 分支 A: 光滑区 (极速模式)
    # ==============================
    if ϕx < hybrid_ϕ1
        for n = 1:5
            @inbounds v1 = U[i-3,j,k,n]; v2 = U[i-2,j,k,n]; v3 = U[i-1,j,k,n]
            @inbounds v4 = U[i  ,j,k,n]; v5 = U[i+1,j,k,n]; v6 = U[i+2,j,k,n]; v7 = U[i+3,j,k,n]
            
            valL = Linear[1]*v1 + Linear[2]*v2 + Linear[3]*v3 + Linear[4]*v4 + Linear[5]*v5 + Linear[6]*v6 + Linear[7]*v7
            
            @inbounds r1 = U[i+4,j,k,n]; r2 = U[i+3,j,k,n]; r3 = U[i+2,j,k,n]
            @inbounds r4 = U[i+1,j,k,n]; r5 = U[i  ,j,k,n]; r6 = U[i-1,j,k,n]; r7 = U[i-2,j,k,n]
            
            valR = Linear[1]*r1 + Linear[2]*r2 + Linear[3]*r3 + Linear[4]*r4 + Linear[5]*r5 + Linear[6]*r6 + Linear[7]*r7

            if n==1; UL_final_1=valL; UR_final_1=valR
            elseif n==2; UL_final_2=valL; UR_final_2=valR
            elseif n==3; UL_final_3=valL; UR_final_3=valR
            elseif n==4; UL_final_4=valL; UR_final_4=valR
            else; UL_final_5=valL; UR_final_5=valR; end
        end

    # ==============================
    # 分支 B: 间断区 (特征分解模式)
    # ==============================
    else 
        WENOϵ1 = 1.0f-10; WENOϵ2 = 1.0f-8
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0
        @inbounds ss = 2.0f0/(S[i+1, j, k] + S[i, j, k])

        for n = 1:5

            # 2b. 投影 U -> V (Component-wise Reconstruction)
            @inbounds V1L = U[i-3,j,k,n]
            @inbounds V2L = U[i-2,j,k,n]
            @inbounds V3L = U[i-1,j,k,n]
            @inbounds V4L = U[i  ,j,k,n]
            @inbounds V5L = U[i+1,j,k,n]
            @inbounds V6L = U[i+2,j,k,n]
            @inbounds V7L = U[i+3,j,k,n]

            @inbounds V1R = U[i+4,j,k,n]
            @inbounds V2R = U[i+3,j,k,n]
            @inbounds V3R = U[i+2,j,k,n]
            @inbounds V4R = U[i+1,j,k,n]
            @inbounds V5R = U[i  ,j,k,n]
            @inbounds V6R = U[i-1,j,k,n]
            @inbounds V7R = U[i-2,j,k,n]

            valL = 0.0f0; valR = 0.0f0

            # 2c. WENO Reconstruction
            if ϕx < hybrid_ϕ2 # WENO7
                q1L = -3.0f0*V1L + 13.0f0*V2L - 23.0f0*V3L + 25.0f0*V4L; q1R = -3.0f0*V1R + 13.0f0*V2R - 23.0f0*V3R + 25.0f0*V4R
                q2L =  1.0f0*V2L -  5.0f0*V3L + 13.0f0*V4L +  3.0f0*V5L; q2R =  1.0f0*V2R -  5.0f0*V3R + 13.0f0*V4R +  3.0f0*V5R
                q3L = -1.0f0*V3L +  7.0f0*V4L +  7.0f0*V5L -  1.0f0*V6L; q3R = -1.0f0*V3R +  7.0f0*V4R +  7.0f0*V5R -  1.0f0*V6R
                q4L =  3.0f0*V4L + 13.0f0*V5L -  5.0f0*V6L +  1.0f0*V7L; q4R =  3.0f0*V4R + 13.0f0*V5R -  5.0f0*V6R +  1.0f0*V7R

                Is1L = V1L*( 547.0f0*V1L - 3882.0f0*V2L + 4642.0f0*V3L - 1854.0f0*V4L) + V2L*( 7043.0f0*V2L -17246.0f0*V3L + 7042.0f0*V4L) + V3L*(11003.0f0*V3L - 9402.0f0*V4L) + V4L*( 2107.0f0*V4L)
                Is2L = V2L*( 267.0f0*V2L - 1642.0f0*V3L + 1602.0f0*V4L -  494.0f0*V5L) + V3L*( 2843.0f0*V3L - 5966.0f0*V4L + 1922.0f0*V5L) + V4L*( 3443.0f0*V4L - 2522.0f0*V5L) + V5L*(  547.0f0*V5L)
                Is3L = V3L*( 547.0f0*V3L - 2522.0f0*V4L + 1922.0f0*V5L -  494.0f0*V6L) + V4L*( 3443.0f0*V4L - 5966.0f0*V5L + 1602.0f0*V6L) + V5L*( 2843.0f0*V5L - 1642.0f0*V6L) + V6L*(  267.0f0*V6L)
                Is4L = V4L*( 2107.0f0*V4L - 9402.0f0*V5L + 7042.0f0*V6L - 1854.0f0*V7L) + V5L*(11003.0f0*V5L -17246.0f0*V6L + 4642.0f0*V7L) + V6L*( 7043.0f0*V6L - 3882.0f0*V7L) + V7L*(  547.0f0*V7L)

                Is1R = V1R*( 547.0f0*V1R - 3882.0f0*V2R + 4642.0f0*V3R - 1854.0f0*V4R) + V2R*( 7043.0f0*V2R -17246.0f0*V3R + 7042.0f0*V4R) + V3R*(11003.0f0*V3R - 9402.0f0*V4R) + V4R*( 2107.0f0*V4R)
                Is2R = V2R*( 267.0f0*V2R - 1642.0f0*V3R + 1602.0f0*V4R -  494.0f0*V5R) + V3R*( 2843.0f0*V3R - 5966.0f0*V4R + 1922.0f0*V5R) + V4R*( 3443.0f0*V4R - 2522.0f0*V5R) + V5R*(  547.0f0*V5R)
                Is3R = V3R*( 547.0f0*V3R - 2522.0f0*V4R + 1922.0f0*V5R -  494.0f0*V6R) + V4R*( 3443.0f0*V4R - 5966.0f0*V5R + 1602.0f0*V6R) + V5R*( 2843.0f0*V5R - 1642.0f0*V6R) + V6R*(  267.0f0*V6R)
                Is4R = V4R*( 2107.0f0*V4R - 9402.0f0*V5R + 7042.0f0*V6R - 1854.0f0*V7R) + V5R*(11003.0f0*V5R -17246.0f0*V6R + 4642.0f0*V7R) + V6R*( 7043.0f0*V6R - 3882.0f0*V7R) + V7R*(  547.0f0*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(t_d1L * t_d1L);  t_d1R = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2L = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(t_d2L * t_d2L); t_d2R = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(t_d2R * t_d2R)
                t_d3L = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(t_d3L * t_d3L); t_d3R = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(t_d3R * t_d3R)
                t_d4L = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(t_d4L * t_d4L);  t_d4R = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(t_d4R * t_d4R)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1

                # ... (valL 和 valR 计算完毕) ...
            
            # ===  WENO5 分支 ===
            elseif ϕx < hybrid_ϕ3 # WENO5
                # 注意：WENO5 只需要 5 个点。
                # V2L (i-2), V3L (i-1), V4L (i), V5L (i+1), V6L (i+2)
                # 对应标准 WENO5 的 v1...v5
                
                # Left Side
                # Beta 1: (13/12)(v1-2v2+v3)^2 + (1/4)(v1-4v2+3v3)^2
                t1 = V2L - 2.0f0*V3L + V4L; t2 = V2L - 4.0f0*V3L + 3.0f0*V4L
                s1L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                
                # Beta 2: (13/12)(v2-2v3+v4)^2 + (1/4)(v2-v4)^2
                t1 = V3L - 2.0f0*V4L + V5L; t2 = V3L - V5L
                s2L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                
                # Beta 3: (13/12)(v3-2v4+v5)^2 + (1/4)(3v3-4v4+v5)^2
                t1 = V4L - 2.0f0*V5L + V6L; t2 = 3.0f0*V4L - 4.0f0*V5L + V6L
                s3L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                # Weights (d0=1/10, d1=6/10, d2=3/10 -> relative 1, 6, 3)
                t_d1L = WENOϵ2 + s1L*ss; α1L = 1.0f0/(t_d1L * t_d1L)
                t_d2L = WENOϵ2 + s2L*ss; α2L = 6.0f0/(t_d2L * t_d2L)
                t_d3L = WENOϵ2 + s3L*ss; α3L = 3.0f0/(t_d3L * t_d3L)
                invsumL = 1.0f0/(α1L+α2L+α3L)

                # Candidates
                v1 = 2.0f0*V2L - 7.0f0*V3L + 11.0f0*V4L
                v2 = -1.0f0*V3L + 5.0f0*V4L + 2.0f0*V5L
                v3 = 2.0f0*V4L + 5.0f0*V5L - 1.0f0*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2 # tmp2 is 1/6

                # Right Side (Symmetric)
                # Use V2R...V6R
                t1 = V2R - 2.0f0*V3R + V4R; t2 = V2R - 4.0f0*V3R + 3.0f0*V4R
                s1R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3R - 2.0f0*V4R + V5R; t2 = V3R - V5R
                s2R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4R - 2.0f0*V5R + V6R; t2 = 3.0f0*V4R - 4.0f0*V5R + V6R
                s3R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2R = WENOϵ2 + s2R*ss; α2R = 6.0f0/(t_d2R * t_d2R)
                t_d3R = WENOϵ2 + s3R*ss; α3R = 3.0f0/(t_d3R * t_d3R)
                invsumR = 1.0f0/(α1R+α2R+α3R)

                v1 = 2.0f0*V2R - 7.0f0*V3R + 11.0f0*V4R
                v2 = -1.0f0*V3R + 5.0f0*V4R + 2.0f0*V5R
                v3 = 2.0f0*V4R + 5.0f0*V5R - 1.0f0*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + 0.5f0*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - 0.5f0*minmod(V4R - V3R, V4R - V5R)
            end
            
            if n == 1;      UL_final_1 = valL; UR_final_1 = valR
            elseif n == 2;  UL_final_2 = valL; UR_final_2 = valR
            elseif n == 3;  UL_final_3 = valL; UR_final_3 = valR
            elseif n == 4;  UL_final_4 = valL; UR_final_4 = valR
            else;           UL_final_5 = valL; UR_final_5 = valR
            end
        end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{5,Float32}(UL_final_1, UL_final_2, UL_final_3, UL_final_4, UL_final_5)
    UR_vec = SVector{5,Float32}(UR_final_1, UR_final_2, UR_final_3, UR_final_4, UR_final_5)
    
    flux_temp = HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)

    @inbounds begin
        Fx[i-NG+1, j-NG, k-NG, 1] = flux_temp[1] * Area
        Fx[i-NG+1, j-NG, k-NG, 2] = flux_temp[2] * Area
        Fx[i-NG+1, j-NG, k-NG, 3] = flux_temp[3] * Area
        Fx[i-NG+1, j-NG, k-NG, 4] = flux_temp[4] * Area
        Fx[i-NG+1, j-NG, k-NG, 5] = flux_temp[5] * Area
    end
    return 
end

function Conser_reconstruct_j(Q, U, ϕ, S, Fy, dηdx, dηdy, dηdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # 1. 边界检查
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG || k < NG+1
        return
    end

    # 2. 几何计算 (使用 eta 度量项，j 和 j+1 界面)
    @inbounds nx = (dηdx[i, j, k] + dηdx[i, j+1, k]) * 0.5f0
    @inbounds ny = (dηdy[i, j, k] + dηdy[i, j+1, k]) * 0.5f0
    @inbounds nz = (dηdz[i, j, k] + dηdz[i, j+1, k]) * 0.5f0
    @inbounds Area = sqrt(nx*nx + ny*ny + nz*nz + 1.0f-20)
    @inbounds inv_len = 1.0f0 / Area
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len

    # 3. 激波传感器 (沿 j 方向)
    @inbounds ϕy = max(ϕ[i, j-2, k], ϕ[i, j-1, k], ϕ[i, j, k], ϕ[i, j+1, k], ϕ[i, j+2, k], ϕ[i, j+3, k])

    # 准备最终累加变量
    UL_final_1 = 0.0f0; UL_final_2 = 0.0f0; UL_final_3 = 0.0f0; UL_final_4 = 0.0f0; UL_final_5 = 0.0f0
    UR_final_1 = 0.0f0; UR_final_2 = 0.0f0; UR_final_3 = 0.0f0; UR_final_4 = 0.0f0; UR_final_5 = 0.0f0

    # ==============================
    # 分支 A: 光滑区
    # ==============================
    if ϕy < hybrid_ϕ1
        for n = 1:5
            # 模版沿 j 方向取值
            @inbounds v1 = U[i,j-3,k,n]; v2 = U[i,j-2,k,n]; v3 = U[i,j-1,k,n]
            @inbounds v4 = U[i,j  ,k,n]; v5 = U[i,j+1,k,n]; v6 = U[i,j+2,k,n]; v7 = U[i,j+3,k,n]
            
            valL = Linear[1]*v1 + Linear[2]*v2 + Linear[3]*v3 + Linear[4]*v4 + Linear[5]*v5 + Linear[6]*v6 + Linear[7]*v7
            
            @inbounds r1 = U[i,j+4,k,n]; r2 = U[i,j+3,k,n]; r3 = U[i,j+2,k,n]
            @inbounds r4 = U[i,j+1,k,n]; r5 = U[i,j  ,k,n]; r6 = U[i,j-1,k,n]; r7 = U[i,j-2,k,n]
            
            valR = Linear[1]*r1 + Linear[2]*r2 + Linear[3]*r3 + Linear[4]*r4 + Linear[5]*r5 + Linear[6]*r6 + Linear[7]*r7

            if n==1;     UL_final_1=valL; UR_final_1=valR
            elseif n==2; UL_final_2=valL; UR_final_2=valR
            elseif n==3; UL_final_3=valL; UR_final_3=valR
            elseif n==4; UL_final_4=valL; UR_final_4=valR
            else;        UL_final_5=valL; UR_final_5=valR; end
        end

    # ==============================
    # 分支 B: 间断区
    # ==============================
    else 
        WENOϵ1 = 1.0f-10; WENOϵ2 = 1.0f-8
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0
        # 平滑因子 ss 沿 j 方向计算
        @inbounds ss = 2.0f0/(S[i, j+1, k] + S[i, j, k])

        for n = 1:5
            # 沿 j 方向读取模板
            @inbounds V1L = U[i,j-3,k,n]
            @inbounds V2L = U[i,j-2,k,n]
            @inbounds V3L = U[i,j-1,k,n]
            @inbounds V4L = U[i,j  ,k,n]
            @inbounds V5L = U[i,j+1,k,n]
            @inbounds V6L = U[i,j+2,k,n]
            @inbounds V7L = U[i,j+3,k,n]

            @inbounds V1R = U[i,j+4,k,n]
            @inbounds V2R = U[i,j+3,k,n]
            @inbounds V3R = U[i,j+2,k,n]
            @inbounds V4R = U[i,j+1,k,n]
            @inbounds V5R = U[i,j  ,k,n]
            @inbounds V6R = U[i,j-1,k,n]
            @inbounds V7R = U[i,j-2,k,n]

            valL = 0.0f0; valR = 0.0f0

            # WENO Reconstruction 逻辑与 i 方向完全一致，仅输入变量不同
            if ϕy < hybrid_ϕ2 # WENO7
                q1L = -3.0f0*V1L + 13.0f0*V2L - 23.0f0*V3L + 25.0f0*V4L; q1R = -3.0f0*V1R + 13.0f0*V2R - 23.0f0*V3R + 25.0f0*V4R
                q2L =  1.0f0*V2L -  5.0f0*V3L + 13.0f0*V4L +  3.0f0*V5L; q2R =  1.0f0*V2R -  5.0f0*V3R + 13.0f0*V4R +  3.0f0*V5R
                q3L = -1.0f0*V3L +  7.0f0*V4L +  7.0f0*V5L -  1.0f0*V6L; q3R = -1.0f0*V3R +  7.0f0*V4R +  7.0f0*V5R -  1.0f0*V6R
                q4L =  3.0f0*V4L + 13.0f0*V5L -  5.0f0*V6L +  1.0f0*V7L; q4R =  3.0f0*V4R + 13.0f0*V5R -  5.0f0*V6R +  1.0f0*V7R

                Is1L = V1L*( 547.0f0*V1L - 3882.0f0*V2L + 4642.0f0*V3L - 1854.0f0*V4L) + V2L*( 7043.0f0*V2L -17246.0f0*V3L + 7042.0f0*V4L) + V3L*(11003.0f0*V3L - 9402.0f0*V4L) + V4L*( 2107.0f0*V4L)
                Is2L = V2L*( 267.0f0*V2L - 1642.0f0*V3L + 1602.0f0*V4L -  494.0f0*V5L) + V3L*( 2843.0f0*V3L - 5966.0f0*V4L + 1922.0f0*V5L) + V4L*( 3443.0f0*V4L - 2522.0f0*V5L) + V5L*(  547.0f0*V5L)
                Is3L = V3L*( 547.0f0*V3L - 2522.0f0*V4L + 1922.0f0*V5L -  494.0f0*V6L) + V4L*( 3443.0f0*V4L - 5966.0f0*V5L + 1602.0f0*V6L) + V5L*( 2843.0f0*V5L - 1642.0f0*V6L) + V6L*(  267.0f0*V6L)
                Is4L = V4L*( 2107.0f0*V4L - 9402.0f0*V5L + 7042.0f0*V6L - 1854.0f0*V7L) + V5L*(11003.0f0*V5L -17246.0f0*V6L + 4642.0f0*V7L) + V6L*( 7043.0f0*V6L - 3882.0f0*V7L) + V7L*(  547.0f0*V7L)

                Is1R = V1R*( 547.0f0*V1R - 3882.0f0*V2R + 4642.0f0*V3R - 1854.0f0*V4R) + V2R*( 7043.0f0*V2R -17246.0f0*V3R + 7042.0f0*V4R) + V3R*(11003.0f0*V3R - 9402.0f0*V4R) + V4R*( 2107.0f0*V4R)
                Is2R = V2R*( 267.0f0*V2R - 1642.0f0*V3R + 1602.0f0*V4R -  494.0f0*V5R) + V3R*( 2843.0f0*V3R - 5966.0f0*V4R + 1922.0f0*V5R) + V4R*( 3443.0f0*V4R - 2522.0f0*V5R) + V5R*(  547.0f0*V5R)
                Is3R = V3R*( 547.0f0*V3R - 2522.0f0*V4R + 1922.0f0*V5R -  494.0f0*V6R) + V4R*( 3443.0f0*V4R - 5966.0f0*V5R + 1602.0f0*V6R) + V5R*( 2843.0f0*V5R - 1642.0f0*V6R) + V6R*(  267.0f0*V6R)
                Is4R = V4R*( 2107.0f0*V4R - 9402.0f0*V5R + 7042.0f0*V6R - 1854.0f0*V7R) + V5R*(11003.0f0*V5R -17246.0f0*V6R + 4642.0f0*V7R) + V6R*( 7043.0f0*V6R - 3882.0f0*V7R) + V7R*(  547.0f0*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(t_d1L * t_d1L);  t_d1R = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2L = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(t_d2L * t_d2L); t_d2R = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(t_d2R * t_d2R)
                t_d3L = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(t_d3L * t_d3L); t_d3R = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(t_d3R * t_d3R)
                t_d4L = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(t_d4L * t_d4L);  t_d4R = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(t_d4R * t_d4R)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1

            elseif ϕy < hybrid_ϕ3 # WENO5
                t1 = V2L - 2.0f0*V3L + V4L; t2 = V2L - 4.0f0*V3L + 3.0f0*V4L
                s1L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3L - 2.0f0*V4L + V5L; t2 = V3L - V5L
                s2L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4L - 2.0f0*V5L + V6L; t2 = 3.0f0*V4L - 4.0f0*V5L + V6L
                s3L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1L = WENOϵ2 + s1L*ss; α1L = 1.0f0/(t_d1L * t_d1L)
                t_d2L = WENOϵ2 + s2L*ss; α2L = 6.0f0/(t_d2L * t_d2L)
                t_d3L = WENOϵ2 + s3L*ss; α3L = 3.0f0/(t_d3L * t_d3L)
                invsumL = 1.0f0/(α1L+α2L+α3L)

                v1 = 2.0f0*V2L - 7.0f0*V3L + 11.0f0*V4L
                v2 = -1.0f0*V3L + 5.0f0*V4L + 2.0f0*V5L
                v3 = 2.0f0*V4L + 5.0f0*V5L - 1.0f0*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2

                t1 = V2R - 2.0f0*V3R + V4R; t2 = V2R - 4.0f0*V3R + 3.0f0*V4R
                s1R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3R - 2.0f0*V4R + V5R; t2 = V3R - V5R
                s2R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4R - 2.0f0*V5R + V6R; t2 = 3.0f0*V4R - 4.0f0*V5R + V6R
                s3R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2R = WENOϵ2 + s2R*ss; α2R = 6.0f0/(t_d2R * t_d2R)
                t_d3R = WENOϵ2 + s3R*ss; α3R = 3.0f0/(t_d3R * t_d3R)
                invsumR = 1.0f0/(α1R+α2R+α3R)

                v1 = 2.0f0*V2R - 7.0f0*V3R + 11.0f0*V4R
                v2 = -1.0f0*V3R + 5.0f0*V4R + 2.0f0*V5R
                v3 = 2.0f0*V4R + 5.0f0*V5R - 1.0f0*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + 0.5f0*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - 0.5f0*minmod(V4R - V3R, V4R - V5R)
            end
            
            if n == 1;      UL_final_1 = valL; UR_final_1 = valR
            elseif n == 2;  UL_final_2 = valL; UR_final_2 = valR
            elseif n == 3;  UL_final_3 = valL; UR_final_3 = valR
            elseif n == 4;  UL_final_4 = valL; UR_final_4 = valR
            else;           UL_final_5 = valL; UR_final_5 = valR
            end
        end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{5,Float32}(UL_final_1, UL_final_2, UL_final_3, UL_final_4, UL_final_5)
    UR_vec = SVector{5,Float32}(UR_final_1, UR_final_2, UR_final_3, UR_final_4, UR_final_5)
    
    flux_temp = HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)

    @inbounds begin
        # 注意：Fy 的写入位置 j 偏移 1
        Fy[i-NG, j-NG+1, k-NG, 1] = UR_final_1#flux_temp[1] * Area
        Fy[i-NG, j-NG+1, k-NG, 2] = UL_final_2#flux_temp[2] * Area
        Fy[i-NG, j-NG+1, k-NG, 3] = UR_final_3#flux_temp[3] * Area
        Fy[i-NG, j-NG+1, k-NG, 4] = UR_final_4#flux_temp[4] * Area
        Fy[i-NG, j-NG+1, k-NG, 5] = UR_final_5#flux_temp[5] * Area
    end
    return 
end

function Conser_reconstruct_k(Q, U, ϕ, S, Fz, dζdx, dζdy, dζdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # 1. 边界检查
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG
        return
    end

    # 2. 几何计算 (使用 zeta 度量项，k 和 k+1 界面)
    @inbounds nx = (dζdx[i, j, k] + dζdx[i, j, k+1]) * 0.5f0
    @inbounds ny = (dζdy[i, j, k] + dζdy[i, j, k+1]) * 0.5f0
    @inbounds nz = (dζdz[i, j, k] + dζdz[i, j, k+1]) * 0.5f0
    @inbounds Area = sqrt(nx*nx + ny*ny + nz*nz + 1.0f-20)
    @inbounds inv_len = 1.0f0 / Area
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len

    # 3. 激波传感器 (沿 k 方向)
    @inbounds ϕz = max(ϕ[i, j, k-2], ϕ[i, j, k-1], ϕ[i, j, k], ϕ[i, j, k+1], ϕ[i, j, k+2], ϕ[i, j, k+3])

    # 准备最终累加变量
    UL_final_1 = 0.0f0; UL_final_2 = 0.0f0; UL_final_3 = 0.0f0; UL_final_4 = 0.0f0; UL_final_5 = 0.0f0
    UR_final_1 = 0.0f0; UR_final_2 = 0.0f0; UR_final_3 = 0.0f0; UR_final_4 = 0.0f0; UR_final_5 = 0.0f0

    # ==============================
    # 分支 A: 光滑区
    # ==============================
    if ϕz < hybrid_ϕ1
        for n = 1:5
            # 模版沿 k 方向取值
            @inbounds v1 = U[i,j,k-3,n]; v2 = U[i,j,k-2,n]; v3 = U[i,j,k-1,n]
            @inbounds v4 = U[i,j,k  ,n]; v5 = U[i,j,k+1,n]; v6 = U[i,j,k+2,n]; v7 = U[i,j,k+3,n]
            
            valL = Linear[1]*v1 + Linear[2]*v2 + Linear[3]*v3 + Linear[4]*v4 + Linear[5]*v5 + Linear[6]*v6 + Linear[7]*v7
            
            @inbounds r1 = U[i,j,k+4,n]; r2 = U[i,j,k+3,n]; r3 = U[i,j,k+2,n]
            @inbounds r4 = U[i,j,k+1,n]; r5 = U[i,j,k  ,n]; r6 = U[i,j,k-1,n]; r7 = U[i,j,k-2,n]
            
            valR = Linear[1]*r1 + Linear[2]*r2 + Linear[3]*r3 + Linear[4]*r4 + Linear[5]*r5 + Linear[6]*r6 + Linear[7]*r7

            if n==1;     UL_final_1=valL; UR_final_1=valR
            elseif n==2; UL_final_2=valL; UR_final_2=valR
            elseif n==3; UL_final_3=valL; UR_final_3=valR
            elseif n==4; UL_final_4=valL; UR_final_4=valR
            else;        UL_final_5=valL; UR_final_5=valR; end
        end

    # ==============================
    # 分支 B: 间断区
    # ==============================
    else 
        WENOϵ1 = 1.0f-10; WENOϵ2 = 1.0f-8
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0
        # 平滑因子 ss 沿 k 方向计算
        @inbounds ss = 2.0f0/(S[i, j, k+1] + S[i, j, k])

        for n = 1:5
            # 沿 k 方向读取模板
            @inbounds V1L = U[i,j,k-3,n]
            @inbounds V2L = U[i,j,k-2,n]
            @inbounds V3L = U[i,j,k-1,n]
            @inbounds V4L = U[i,j,k  ,n]
            @inbounds V5L = U[i,j,k+1,n]
            @inbounds V6L = U[i,j,k+2,n]
            @inbounds V7L = U[i,j,k+3,n]

            @inbounds V1R = U[i,j,k+4,n]
            @inbounds V2R = U[i,j,k+3,n]
            @inbounds V3R = U[i,j,k+2,n]
            @inbounds V4R = U[i,j,k+1,n]
            @inbounds V5R = U[i,j,k  ,n]
            @inbounds V6R = U[i,j,k-1,n]
            @inbounds V7R = U[i,j,k-2,n]

            valL = 0.0f0; valR = 0.0f0

            if ϕz < hybrid_ϕ2 # WENO7
                q1L = -3.0f0*V1L + 13.0f0*V2L - 23.0f0*V3L + 25.0f0*V4L; q1R = -3.0f0*V1R + 13.0f0*V2R - 23.0f0*V3R + 25.0f0*V4R
                q2L =  1.0f0*V2L -  5.0f0*V3L + 13.0f0*V4L +  3.0f0*V5L; q2R =  1.0f0*V2R -  5.0f0*V3R + 13.0f0*V4R +  3.0f0*V5R
                q3L = -1.0f0*V3L +  7.0f0*V4L +  7.0f0*V5L -  1.0f0*V6L; q3R = -1.0f0*V3R +  7.0f0*V4R +  7.0f0*V5R -  1.0f0*V6R
                q4L =  3.0f0*V4L + 13.0f0*V5L -  5.0f0*V6L +  1.0f0*V7L; q4R =  3.0f0*V4R + 13.0f0*V5R -  5.0f0*V6R +  1.0f0*V7R

                Is1L = V1L*( 547.0f0*V1L - 3882.0f0*V2L + 4642.0f0*V3L - 1854.0f0*V4L) + V2L*( 7043.0f0*V2L -17246.0f0*V3L + 7042.0f0*V4L) + V3L*(11003.0f0*V3L - 9402.0f0*V4L) + V4L*( 2107.0f0*V4L)
                Is2L = V2L*( 267.0f0*V2L - 1642.0f0*V3L + 1602.0f0*V4L -  494.0f0*V5L) + V3L*( 2843.0f0*V3L - 5966.0f0*V4L + 1922.0f0*V5L) + V4L*( 3443.0f0*V4L - 2522.0f0*V5L) + V5L*(  547.0f0*V5L)
                Is3L = V3L*( 547.0f0*V3L - 2522.0f0*V4L + 1922.0f0*V5L -  494.0f0*V6L) + V4L*( 3443.0f0*V4L - 5966.0f0*V5L + 1602.0f0*V6L) + V5L*( 2843.0f0*V5L - 1642.0f0*V6L) + V6L*(  267.0f0*V6L)
                Is4L = V4L*( 2107.0f0*V4L - 9402.0f0*V5L + 7042.0f0*V6L - 1854.0f0*V7L) + V5L*(11003.0f0*V5L -17246.0f0*V6L + 4642.0f0*V7L) + V6L*( 7043.0f0*V6L - 3882.0f0*V7L) + V7L*(  547.0f0*V7L)

                Is1R = V1R*( 547.0f0*V1R - 3882.0f0*V2R + 4642.0f0*V3R - 1854.0f0*V4R) + V2R*( 7043.0f0*V2R -17246.0f0*V3R + 7042.0f0*V4R) + V3R*(11003.0f0*V3R - 9402.0f0*V4R) + V4R*( 2107.0f0*V4R)
                Is2R = V2R*( 267.0f0*V2R - 1642.0f0*V3R + 1602.0f0*V4R -  494.0f0*V5R) + V3R*( 2843.0f0*V3R - 5966.0f0*V4R + 1922.0f0*V5R) + V4R*( 3443.0f0*V4R - 2522.0f0*V5R) + V5R*(  547.0f0*V5R)
                Is3R = V3R*( 547.0f0*V3R - 2522.0f0*V4R + 1922.0f0*V5R -  494.0f0*V6R) + V4R*( 3443.0f0*V4R - 5966.0f0*V5R + 1602.0f0*V6R) + V5R*( 2843.0f0*V5R - 1642.0f0*V6R) + V6R*(  267.0f0*V6R)
                Is4R = V4R*( 2107.0f0*V4R - 9402.0f0*V5R + 7042.0f0*V6R - 1854.0f0*V7R) + V5R*(11003.0f0*V5R -17246.0f0*V6R + 4642.0f0*V7R) + V6R*( 7043.0f0*V6R - 3882.0f0*V7R) + V7R*(  547.0f0*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(t_d1L * t_d1L);  t_d1R = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2L = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(t_d2L * t_d2L); t_d2R = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(t_d2R * t_d2R)
                t_d3L = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(t_d3L * t_d3L); t_d3R = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(t_d3R * t_d3R)
                t_d4L = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(t_d4L * t_d4L);  t_d4R = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(t_d4R * t_d4R)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1

            elseif ϕz < hybrid_ϕ3 # WENO5
                t1 = V2L - 2.0f0*V3L + V4L; t2 = V2L - 4.0f0*V3L + 3.0f0*V4L
                s1L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3L - 2.0f0*V4L + V5L; t2 = V3L - V5L
                s2L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4L - 2.0f0*V5L + V6L; t2 = 3.0f0*V4L - 4.0f0*V5L + V6L
                s3L = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1L = WENOϵ2 + s1L*ss; α1L = 1.0f0/(t_d1L * t_d1L)
                t_d2L = WENOϵ2 + s2L*ss; α2L = 6.0f0/(t_d2L * t_d2L)
                t_d3L = WENOϵ2 + s3L*ss; α3L = 3.0f0/(t_d3L * t_d3L)
                invsumL = 1.0f0/(α1L+α2L+α3L)

                v1 = 2.0f0*V2L - 7.0f0*V3L + 11.0f0*V4L
                v2 = -1.0f0*V3L + 5.0f0*V4L + 2.0f0*V5L
                v3 = 2.0f0*V4L + 5.0f0*V5L - 1.0f0*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2

                t1 = V2R - 2.0f0*V3R + V4R; t2 = V2R - 4.0f0*V3R + 3.0f0*V4R
                s1R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V3R - 2.0f0*V4R + V5R; t2 = V3R - V5R
                s2R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2
                t1 = V4R - 2.0f0*V5R + V6R; t2 = 3.0f0*V4R - 4.0f0*V5R + V6R
                s3R = 13.0f0 * t1*t1 + 3.0f0 * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; α1R = 1.0f0/(t_d1R * t_d1R)
                t_d2R = WENOϵ2 + s2R*ss; α2R = 6.0f0/(t_d2R * t_d2R)
                t_d3R = WENOϵ2 + s3R*ss; α3R = 3.0f0/(t_d3R * t_d3R)
                invsumR = 1.0f0/(α1R+α2R+α3R)

                v1 = 2.0f0*V2R - 7.0f0*V3R + 11.0f0*V4R
                v2 = -1.0f0*V3R + 5.0f0*V4R + 2.0f0*V5R
                v3 = 2.0f0*V4R + 5.0f0*V5R - 1.0f0*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + 0.5f0*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - 0.5f0*minmod(V4R - V3R, V4R - V5R)
            end
            
            if n == 1;      UL_final_1 = valL; UR_final_1 = valR
            elseif n == 2;  UL_final_2 = valL; UR_final_2 = valR
            elseif n == 3;  UL_final_3 = valL; UR_final_3 = valR
            elseif n == 4;  UL_final_4 = valL; UR_final_4 = valR
            else;           UL_final_5 = valL; UR_final_5 = valR
            end
        end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{5,Float32}(UL_final_1, UL_final_2, UL_final_3, UL_final_4, UL_final_5)
    UR_vec = SVector{5,Float32}(UR_final_1, UR_final_2, UR_final_3, UR_final_4, UR_final_5)
    
    flux_temp = HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)

    @inbounds begin
        # 注意：Fz 的写入位置 k 偏移 1
        Fz[i-NG, j-NG, k-NG+1, 1] = UR_final_1#flux_temp[1] * Area
        Fz[i-NG, j-NG, k-NG+1, 2] = UL_final_2#flux_temp[2] * Area
        Fz[i-NG, j-NG, k-NG+1, 3] = UR_final_3#flux_temp[3] * Area
        Fz[i-NG, j-NG, k-NG+1, 4] = UR_final_4#flux_temp[4] * Area
        Fz[i-NG, j-NG, k-NG+1, 5] = UR_final_5#flux_temp[5] * Area
    end
    return 
end

@inline function HLLC_Flux(UL, UR, nx, ny, nz)

    # 1. 准备物理量 (Primitive Variables)
    # ------------------------------------------------------------------
    # Left State
    ρL = UL[1]
    inv_ρL = 1 / ρL
    uL = UL[2] * inv_ρL
    vL = UL[3] * inv_ρL
    wL = UL[4] * inv_ρL
    EL = UL[5]
    pL = (γ - 1) * (EL - 0.5f0 * ρL * (uL^2 + vL^2 + wL^2))
    cL = sqrt(abs(γ * pL * inv_ρL))
    qL = uL * nx + vL * ny + wL * nz  # 法向速度

    # Right State
    ρR = UR[1]
    inv_ρR = 1 / ρR
    uR = UR[2] * inv_ρR
    vR = UR[3] * inv_ρR
    wR = UR[4] * inv_ρR
    ER = UR[5]
    pR = (γ - 1) * (ER - 0.5f0 * ρR * (uR^2 + vR^2 + wR^2))
    cR = sqrt(abs(γ * pR * inv_ρR))
    qR = uR * nx + vR * ny + wR * nz  # 法向速度

    # 2. 计算波速 (Wave Speeds Estimate using Roe Averages)
    # ------------------------------------------------------------------
    # Roe 平均
    sqρL = sqrt(ρL)
    sqρR = sqrt(ρR)
    inv_sqρ = 1.0f0 / (sqρL + sqρR)
    
    # Roe 速度
    u_roe = (sqρL * uL + sqρR * uR) * inv_sqρ
    v_roe = (sqρL * vL + sqρR * vR) * inv_sqρ
    w_roe = (sqρL * wL + sqρR * wR) * inv_sqρ
    q_roe = u_roe * nx + v_roe * ny + w_roe * nz
    
    # Roe 焓
    HL = (EL + pL) * inv_ρL
    HR = (ER + pR) * inv_ρR
    H_roe = (sqρL * HL + sqρR * HR) * inv_sqρ
    
    # Roe 声速
    v2_roe = u_roe^2 + v_roe^2 + w_roe^2
    c_roe = sqrt(abs((γ - 1) * (H_roe - 0.5f0 * v2_roe)))

    # 估计左右波速 (SL, SR)
    SL = min(qL - cL, q_roe - c_roe)
    SR = max(qR + cR, q_roe + c_roe)

    # 3. 计算接触间断波速 (Contact Wave Speed S*)
    # ------------------------------------------------------------------
    # HLLC 对 S* 的定义
    S_star = (pR - pL + ρL * qL * (SL - qL) - ρR * qR * (SR - qR)) / 
             (ρL * (SL - qL) - ρR * (SR - qR) + 1f-12) # 加极小量防除零

    # 4. 根据波速位置选择通量 (Logic Branching)
    # ------------------------------------------------------------------
    if SL >= 0
        # Case 1: Supersonic Flow to Right (L State)
        # 直接返回 FL = F(UL)
        flux1 = ρL * qL
        flux2 = ρL * uL * qL + pL * nx
        flux3 = ρL * vL * qL + pL * ny
        flux4 = ρL * wL * qL + pL * nz
        flux5 = (EL + pL) * qL
        return SVector{5, Float32}(flux1, flux2, flux3, flux4, flux5)

    elseif SR <= 0
        # Case 2: Supersonic Flow to Left (R State)
        # 直接返回 FR = F(UR)
        flux1 = ρR * qR
        flux2 = ρR * uR * qR + pR * nx
        flux3 = ρR * vR * qR + pR * ny
        flux4 = ρR * wR * qR + pR * nz
        flux5 = (ER + pR) * qR
        return SVector{5, Float32}(flux1, flux2, flux3, flux4, flux5)
        
    elseif SL < 0 && S_star >= 0
        # Case 3: Subsonic Left Star Region (HLLC Flux L*)
        # F*_L = F_L + SL * (U*_L - U_L)
        
        # 计算 F_L
        FL1 = ρL * qL
        FL2 = ρL * uL * qL + pL * nx
        FL3 = ρL * vL * qL + pL * ny
        FL4 = ρL * wL * qL + pL * nz
        FL5 = (EL + pL) * qL
        
        # 计算 U*_L (HLLC State)
        factor = ρL * (SL - qL) / (SL - S_star)
        
        Us1 = factor * 1.0f0
        Us2 = factor * (uL + (S_star - qL) * nx)
        Us3 = factor * (vL + (S_star - qL) * ny)
        Us4 = factor * (wL + (S_star - qL) * nz)
        Us5 = factor * (EL * inv_ρL + (S_star - qL) * (S_star + pL / (ρL * (SL - qL))))
        return SVector{5, Float32}(
            FL1 + SL * (Us1 - UL[1]),
            FL2 + SL * (Us2 - UL[2]),
            FL3 + SL * (Us3 - UL[3]),
            FL4 + SL * (Us4 - UL[4]),
            FL5 + SL * (Us5 - UL[5])
        )

    else # S_star < 0 && SR > 0
        # Case 4: Subsonic Right Star Region (HLLC Flux R*)
        # F*_R = F_R + SR * (U*_R - U_R)
        
        # 计算 F_R
        FR1 = ρR * qR
        FR2 = ρR * uR * qR + pR * nx
        FR3 = ρR * vR * qR + pR * ny
        FR4 = ρR * wR * qR + pR * nz
        FR5 = (ER + pR) * qR
        
        # 计算 U*_R (HLLC State)
        factor = ρR * (SR - qR) / (SR - S_star)
        
        Us1 = factor * 1.0f0
        Us2 = factor * (uR + (S_star - qR) * nx)
        Us3 = factor * (vR + (S_star - qR) * ny)
        Us4 = factor * (wR + (S_star - qR) * nz)
        Us5 = factor * (ER * inv_ρR + (S_star - qR) * (S_star + pR / (ρR * (SR - qR))))
        return SVector{5, Float32}(
            FR1 + SR * (Us1 - UR[1]),
            FR2 + SR * (Us2 - UR[2]),
            FR3 + SR * (Us3 - UR[3]),
            FR4 + SR * (Us4 - UR[4]),
            FR5 + SR * (Us5 - UR[5])
        )
    end
end