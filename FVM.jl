include("schemes.jl")

function Eigen_reconstruct_i(Q, U, ϕ, S, Fx, dξdx, dξdy, dξdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG || j < NG+1 || k < NG+1
        return
    end

    @inbounds nx::Float32 = (dξdx[i, j, k] + dξdx[i+1, j, k]) * 0.5f0
    @inbounds ny::Float32 = (dξdy[i, j, k] + dξdy[i+1, j, k]) * 0.5f0
    @inbounds nz::Float32 = (dξdz[i, j, k] + dξdz[i+1, j, k]) * 0.5f0
    @inbounds inv_len = 1.0f0 / sqrt(nx^2 + ny^2 + nz^2 + 1e-12f0)
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len

    # left state reconstruction at i+1/2
    @inbounds begin
        u11 = U[i-3, j, k, 1]; u12 = U[i-2, j, k, 1]; u13 = U[i-1, j, k, 1]; u14 = U[i, j, k, 1]; u15 = U[i+1, j, k, 1]; u16 = U[i+2, j, k, 1]; u17 = U[i+3, j, k, 1]
        u21 = U[i-3, j, k, 2]; u22 = U[i-2, j, k, 2]; u23 = U[i-1, j, k, 2]; u24 = U[i, j, k, 2]; u25 = U[i+1, j, k, 2]; u26 = U[i+2, j, k, 2]; u27 = U[i+3, j, k, 2]
        u31 = U[i-3, j, k, 3]; u32 = U[i-2, j, k, 3]; u33 = U[i-1, j, k, 3]; u34 = U[i, j, k, 3]; u35 = U[i+1, j, k, 3]; u36 = U[i+2, j, k, 3]; u37 = U[i+3, j, k, 3]
        u41 = U[i-3, j, k, 4]; u42 = U[i-2, j, k, 4]; u43 = U[i-1, j, k, 4]; u44 = U[i, j, k, 4]; u45 = U[i+1, j, k, 4]; u46 = U[i+2, j, k, 4]; u47 = U[i+3, j, k, 4]
        u51 = U[i-3, j, k, 5]; u52 = U[i-2, j, k, 5]; u53 = U[i-1, j, k, 5]; u54 = U[i, j, k, 5]; u55 = U[i+1, j, k, 5]; u56 = U[i+2, j, k, 5]; u57 = U[i+3, j, k, 5]
    end
    UL = SMatrix{5,7,Float32}(
        u11, u21, u31, u41, u51,
        u12, u22, u32, u42, u52,
        u13, u23, u33, u43, u53,
        u14, u24, u34, u44, u54,
        u15, u25, u35, u45, u55,
        u16, u26, u36, u46, u56,
        u17, u27, u37, u47, u57
    )
    # right state reconstruction at i+1/2
    @inbounds begin
        u11 = U[i+4, j, k, 1]; u12 = U[i+3, j, k, 1]; u13 = U[i+2, j, k, 1]; u14 = U[i+1, j, k, 1]; u15 = U[i, j, k, 1]; u16 = U[i-1, j, k, 1]; u17 = U[i-2, j, k, 1]
        u21 = U[i+4, j, k, 2]; u22 = U[i+3, j, k, 2]; u23 = U[i+2, j, k, 2]; u24 = U[i+1, j, k, 2]; u25 = U[i, j, k, 2]; u26 = U[i-1, j, k, 2]; u27 = U[i-2, j, k, 2]
        u31 = U[i+4, j, k, 3]; u32 = U[i+3, j, k, 3]; u33 = U[i+2, j, k, 3]; u34 = U[i+1, j, k, 3]; u35 = U[i, j, k, 3]; u36 = U[i-1, j, k, 3]; u37 = U[i-2, j, k, 3]
        u41 = U[i+4, j, k, 4]; u42 = U[i+3, j, k, 4]; u43 = U[i+2, j, k, 4]; u44 = U[i+1, j, k, 4]; u45 = U[i, j, k, 4]; u46 = U[i-1, j, k, 4]; u47 = U[i-2, j, k, 4]
        u51 = U[i+4, j, k, 5]; u52 = U[i+3, j, k, 5]; u53 = U[i+2, j, k, 5]; u54 = U[i+1, j, k, 5]; u55 = U[i, j, k, 5]; u56 = U[i-1, j, k, 5]; u57 = U[i-2, j, k, 5]
    end
    UR = SMatrix{5,7,Float32}(
        u11, u21, u31, u41, u51,
        u12, u22, u32, u42, u52,
        u13, u23, u33, u43, u53,
        u14, u24, u34, u44, u54,
        u15, u25, u35, u45, u55,
        u16, u26, u36, u46, u56,
        u17, u27, u37, u47, u57
    )
    UL_interp = MVector{5,Float32}(undef)
    UR_interp = MVector{5,Float32}(undef)

    # Jameson sensor
    @inbounds ϕx = max(ϕ[i-2, j, k], 
                    ϕ[i-1, j, k], 
                    ϕ[i  , j, k], 
                    ϕ[i+1, j, k], 
                    ϕ[i+2, j, k], 
                    ϕ[i+3, j, k])
    
    if ϕx < hybrid_ϕ1   # smooth region: use linear reconstruction and primitive variables
        c_vec = SVector{7,Float32}(Linear[1], Linear[2], Linear[3], Linear[4], Linear[5], Linear[6], Linear[7])
        UL_interp = UL * c_vec              # 5×1
        UR_interp = UR * c_vec              # 5×1
    else # discontinuous region: use characteristic decomposition and nonlinear reconstruction

        @inbounds if abs(nz) <= abs(ny)
            @inbounds den::Float32 = sqrt(nx^2 + ny^2 + 1e-12f0)
            @inbounds lx::Float32 = -ny / den
            @inbounds ly::Float32 =  nx / den
            @inbounds lz::Float32 =  0.0f0
        else
            @inbounds den::Float32 = sqrt(nx^2 + nz^2 + 1e-12f0)
            @inbounds lx::Float32 = -nz / den
            @inbounds ly::Float32 =  0.0f0
            @inbounds lz::Float32 =  nx / den
        end

        @inbounds mx::Float32 = ny * lz - nz * ly
        @inbounds my::Float32 = nz * lx - nx * lz
        @inbounds mz::Float32 = nx * ly - ny * lx

        # roe averages at i+1/2
        @inbounds ρ = sqrt(Q[i, j, k, 1] * Q[i+1, j, k, 1])
        @inbounds u = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 2] + sqrt(Q[i+1, j, k, 1]) * Q[i+1, j, k, 2]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        @inbounds v = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 3] + sqrt(Q[i+1, j, k, 1]) * Q[i+1, j, k, 3]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        @inbounds w = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 4] + sqrt(Q[i+1, j, k, 1]) * Q[i+1, j, k, 4]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        @inbounds HL = γ/(γ-1f0)*Q[i, j, k, 5]/Q[i, j, k, 1] + 0.5f0*(Q[i, j, k, 2]^2 + Q[i, j, k, 3]^2 + Q[i, j, k, 4]^2)
        @inbounds HR = γ/(γ-1f0)*Q[i+1, j, k, 5]/Q[i+1, j, k, 1] + 0.5f0*(Q[i+1, j, k, 2]^2 + Q[i+1, j, k, 3]^2 + Q[i+1, j, k, 4]^2)
        @inbounds H = (sqrt(Q[i, j, k, 1]) * HL + sqrt(Q[i+1, j, k, 1]) * HR) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i+1, j, k, 1]))
        @inbounds v2 = 0.5f0*(u^2 + v^2 + w^2)
        @inbounds c = sqrt((γ-1f0)*(H - v2))
        @inbounds un = u*nx + v*ny + w*nz
        @inbounds ul = u*lx + v*ly + w*lz
        @inbounds um = u*mx + v*my + w*mz
        @inbounds K = γ-1f0

        @inbounds begin
            invc = 1f0/c; invc2 = invc*invc
            Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
            Kv2 = K*v2*invc2; Kc2 = K*invc2
            un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
            half = 0.5f0; mhalf = -0.5f0
        end
        @inbounds Ln11 = half*(Kv2 + un_invc);  Ln12 = mhalf*(Ku + nx_invc); Ln13 = mhalf*(Kv + ny_invc); Ln14 = mhalf*(Kw + nz_invc); Ln15 = half*Kc2
        @inbounds Ln21 = 1f0 - Kv2;             Ln22 = Ku;                   Ln23 = Kv;                   Ln24 = Kw;                   Ln25 = -Kc2
        @inbounds Ln31 = half*(Kv2 - un_invc);  Ln32 = mhalf*(Ku - nx_invc); Ln33 = mhalf*(Kv - ny_invc); Ln34 = mhalf*(Kw - nz_invc); Ln35 = half*Kc2
        @inbounds Ln41 = -ul;                   Ln42 = lx;                 Ln43 = ly;                 Ln44 = lz;                 Ln45 = 0f0
        @inbounds Ln51 = -um;                   Ln52 = mx;                 Ln53 = my;                 Ln54 = mz;                 Ln55 = 0f0

        @inbounds Rn11 = 1f0;            Rn12 = 1f0;      Rn13 = 1f0;            Rn14 = 0f0;      Rn15 = 0f0
        @inbounds Rn21 = u - nx*c;     Rn22 = u;        Rn23 = u + nx*c;     Rn24 = lx;     Rn25 = mx
        @inbounds Rn31 = v - ny*c;     Rn32 = v;        Rn33 = v + ny*c;     Rn34 = ly;     Rn35 = my
        @inbounds Rn41 = w - nz*c;     Rn42 = w;        Rn43 = w + nz*c;     Rn44 = lz;     Rn45 = mz
        @inbounds Rn51 = H - un*c;     Rn52 = v2;       Rn53 = H + un*c;     Rn54 = ul;     Rn55 = um

        # per-thread small static matrices to avoid GPU heap allocation
        # requires: using StaticArrays
        L = SMatrix{5,5,Float32}(
            Ln11, Ln21, Ln31, Ln41, Ln51,
            Ln12, Ln22, Ln32, Ln42, Ln52,
            Ln13, Ln23, Ln33, Ln43, Ln53,
            Ln14, Ln24, Ln34, Ln44, Ln54,
            Ln15, Ln25, Ln35, Ln45, Ln55
        )
        R = SMatrix{5,5,Float32}(
            Rn11, Rn21, Rn31, Rn41, Rn51,
            Rn12, Rn22, Rn32, Rn42, Rn52,
            Rn13, Rn23, Rn33, Rn43, Rn53,
            Rn14, Rn24, Rn34, Rn44, Rn54,
            Rn15, Rn25, Rn35, Rn45, Rn55
        )

        tmp1::Float32 = 1/12f0
        tmp2::Float32 = 1/6f0

        WENOϵ1::Float32 = 1e-20
        WENOϵ2::Float32 = 1f-16

        @inbounds ss::Float32 = 2/(S[i+1, j, k] + S[i, j, k]) 

        WL = L * UL  # 5×7 per-thread array; WL[r,c] corresponds to old WL{r}{c}
        WR = L * UR  # 5×7 per-thread array
        
        WL_interp = MVector{5,Float32}(undef)
        WR_interp = MVector{5,Float32}(undef)
        if ϕx < hybrid_ϕ2   # WENO region: use WENO reconstruction and characteristic variables
            for n = 1:NV
                @inbounds V1L = WL[n, 1]; V1R = WR[n, 1];
                @inbounds V2L = WL[n, 2]; V2R = WR[n, 2];
                @inbounds V3L = WL[n, 3]; V3R = WR[n, 3];
                @inbounds V4L = WL[n, 4]; V4R = WR[n, 4];
                @inbounds V5L = WL[n, 5]; V5R = WR[n, 5];
                @inbounds V6L = WL[n, 6]; V6R = WR[n, 6];
                @inbounds V7L = WL[n, 7]; V7R = WR[n, 7];
        
                # polinomia
                q1L = -3V1L+13V2L-23V3L+25V4L   ; q1R = -3V1R+13V2R-23V3R+25V4R   
                q2L = V2L-5V3L+13V4L+3V5L       ; q2R = V2R-5V3R+13V4R+3V5R
                q3L = -V3L+7V4L+7V5L-V6L        ; q3R = -V3R+7V4R+7V5R-V6R
                q4L = 3V4L+13V5L-5V6L+V7L       ; q4R = 3V4R+13V5R-5V6R+V7R
        
                # smoothness index
                Is1L = V1L*( 547V1L - 3882V2L + 4642V3L - 1854V4L) + 
                    V2L*(          7043V2L -17246V3L + 7042V4L) +
                    V3L*(                   11003V3L - 9402V4L) +
                    V4L*(                              2107V4L)
                Is2L = V2L*( 267V2L - 1642V3L + 1602V4L -  494V5L) +
                    V3L*(          2843V3L - 5966V4L + 1922V5L) +
                    V4L*(                    3443V4L - 2522V5L) +
                    V5L*(                               547V5L)
                Is3L = V3L*( 547V3L - 2522V4L + 1922V5L -  494V6L) +
                    V4L*(          3443V4L - 5966V5L + 1602V6L) +
                    V5L*(                    2843V5L - 1642V6L) +
                    V6L*(                               267V6L)
                Is4L = V4L*(2107V4L - 9402V5L + 7042V6L - 1854V7L) +
                    V5L*(         11003V5L -17246V6L + 4642V7L) +
                    V6L*(                    7043V6L - 3882V7L) +
                    V7L*(                               547V7L)
        
                Is1R = V1R*( 547V1R - 3882V2R + 4642V3R - 1854V4R) + 
                    V2R*(          7043V2R -17246V3R + 7042V4R) +
                    V3R*(                   11003V3R - 9402V4R) +
                    V4R*(                              2107V4R)
                Is2R = V2R*( 267V2R - 1642V3R + 1602V4R -  494V5R) +
                    V3R*(          2843V3R - 5966V4R + 1922V5R) +
                    V4R*(                    3443V4R - 2522V5R) +
                    V5R*(                               547V5R)
                Is3R = V3R*( 547V3R - 2522V4R + 1922V5R -  494V6R) +
                    V4R*(          3443V4R - 5966V5R + 1602V6R) +
                    V5R*(                    2843V5R - 1642V6R) +
                    V6R*(                               267V6R)
                # alpha
                α1L = 1/(WENOϵ1+Is1L*ss)^2  ; α1R = 1/(WENOϵ1+Is1R*ss)^2  
                α2L = 12/(WENOϵ1+Is2L*ss)^2 ; α2R = 12/(WENOϵ1+Is2R*ss)^2 ;
                α3L = 18/(WENOϵ1+Is3L*ss)^2 ; α3R = 18/(WENOϵ1+Is3R*ss)^2 ;
                α4L = 4/(WENOϵ1+Is4L*ss)^2  ; α4R = 4/(WENOϵ1+Is4R*ss)^2
        
                invsumL = 1/(α1L+α2L+α3L+α4L) ; invsumR = 1/(α1R+α2R+α3R+α4R) 
        
                WL_interp[n] = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                WR_interp[n] = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1
            end
            UL_interp = R * WL_interp              # 5×1
            UR_interp = R * WR_interp              # 5×1
        elseif ϕx < hybrid_ϕ3   # extreme WENO region: use WENO reconstruction and characteristic variables with less stencils
            for n = 1:NV
                @inbounds V1L = WL[n, 2]; V1R = WR[n, 2];
                @inbounds V2L = WL[n, 3]; V2R = WR[n, 3];
                @inbounds V3L = WL[n, 4]; V3R = WR[n, 4];
                @inbounds V4L = WL[n, 5]; V4R = WR[n, 5];
                @inbounds V5L = WL[n, 6]; V5R = WR[n, 6];
                # FP
                s11L = 13*(V1L-2*V2L+V3L)^2 + 3*(V1L-4*V2L+3*V3L)^2 ; s11R = 13*(V1R-2*V2R+V3R)^2 + 3*(V1R-4*V2R+3*V3R)^2 
                s22L = 13*(V2L-2*V3L+V4L)^2 + 3*(V2L-V4L)^2         ; s22R = 13*(V2R-2*V3R+V4R)^2 + 3*(V2R-V4R)^2         
                s33L = 13*(V3L-2*V4L+V5L)^2 + 3*(3*V3L-4*V4L+V5L)^2 ; s33R = 13*(V3R-2*V4R+V5R)^2 + 3*(3*V3R-4*V4R+V5R)^2 

                s11L = 1/(WENOϵ2+s11L*ss)^2 ; s11R = 1/(WENOϵ2+s11R*ss)^2
                s22L = 6/(WENOϵ2+s22L*ss)^2 ; s22R = 6/(WENOϵ2+s22R*ss)^2
                s33L = 3/(WENOϵ2+s33L*ss)^2 ; s33R = 3/(WENOϵ2+s33R*ss)^2

                invsumL = 1/(s11L+s22L+s33L); invsumR = 1/(s11R+s22R+s33R);

                v1L = 2*V1L-7*V2L+11*V3L    ; v1R = 2*V1R-7*V2R+11*V3R
                v2L = -V2L+5*V3L+2*V4L      ; v2R = -V2R+5*V3R+2*V4R
                v3L = 2*V3L+5*V4L-V5L       ; v3R = 2*V3R+5*V4R-V5R
                WL_interp[n] = invsumL*(s11L*v1L+s22L*v2L+s33L*v3L) * tmp2
                WR_interp[n] = invsumR*(s11R*v1R+s22R*v2R+s33R*v3R) * tmp2
            end
            UL_interp = R * WL_interp              # 5×1
            UR_interp = R * WR_interp              # 5×1
        else    # very extreme case: use minmod limiter in characteristic variables
            for n = 1:NV
                @inbounds WL_interp[n] = WL[n, 4] + 0.5f0*minmod(WL[n, 4] - WL[n, 3], 
                                                            WL[n, 5] - WL[n, 4])
                @inbounds WR_interp[n] = WR[n, 4] - 0.5f0*minmod(WR[n, 3] - WR[n, 4], 
                                                            WR[n, 4] - WR[n, 5])
            end 
            UL_interp = R * WL_interp              # 5×1
            UR_interp = R * WR_interp              # 5×1
        end
    end
    Fx[i, j, k, :] = HLLC_Flux(UL_interp, UR_interp, nx, ny, nz)

end

function Eigen_reconstruct_j(Q, U, ϕ, S, Fy, dηdx, dηdy, dηdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # 边界检查：注意这里主要防止 j 方向越界
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG || k < NG+1
        return
    end
    
    @inbounds nx::Float32 = (dηdx[i, j, k] + dηdx[i, j+1, k]) * 0.5f0
    @inbounds ny::Float32 = (dηdy[i, j, k] + dηdy[i, j+1, k]) * 0.5f0
    @inbounds nz::Float32 = (dηdz[i, j, k] + dηdz[i, j+1, k]) * 0.5f0
    @inbounds inv_len = 1.0f0 / sqrt(nx*nx + ny*ny + nz*nz + 1e-12f0)
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len

    # ==========================================
    # 1. Load Stencil (J-direction shift)
    # ==========================================
    # left state reconstruction at j+1/2
    @inbounds begin
        u11 = U[i, j-3, k, 1]; u12 = U[i, j-2, k, 1]; u13 = U[i, j-1, k, 1]; u14 = U[i, j, k, 1]; u15 = U[i, j+1, k, 1]; u16 = U[i, j+2, k, 1]; u17 = U[i, j+3, k, 1]
        u21 = U[i, j-3, k, 2]; u22 = U[i, j-2, k, 2]; u23 = U[i, j-1, k, 2]; u24 = U[i, j, k, 2]; u25 = U[i, j+1, k, 2]; u26 = U[i, j+2, k, 2]; u27 = U[i, j+3, k, 2]
        u31 = U[i, j-3, k, 3]; u32 = U[i, j-2, k, 3]; u33 = U[i, j-1, k, 3]; u34 = U[i, j, k, 3]; u35 = U[i, j+1, k, 3]; u36 = U[i, j+2, k, 3]; u37 = U[i, j+3, k, 3]
        u41 = U[i, j-3, k, 4]; u42 = U[i, j-2, k, 4]; u43 = U[i, j-1, k, 4]; u44 = U[i, j, k, 4]; u45 = U[i, j+1, k, 4]; u46 = U[i, j+2, k, 4]; u47 = U[i, j+3, k, 4]
        u51 = U[i, j-3, k, 5]; u52 = U[i, j-2, k, 5]; u53 = U[i, j-1, k, 5]; u54 = U[i, j, k, 5]; u55 = U[i, j+1, k, 5]; u56 = U[i, j+2, k, 5]; u57 = U[i, j+3, k, 5]
    end
    UL = SMatrix{5,7,Float32}(
        u11, u21, u31, u41, u51,
        u12, u22, u32, u42, u52,
        u13, u23, u33, u43, u53,
        u14, u24, u34, u44, u54,
        u15, u25, u35, u45, u55,
        u16, u26, u36, u46, u56,
        u17, u27, u37, u47, u57
    )

    # right state reconstruction at j+1/2 (Mirrored in J)
    @inbounds begin
        u11 = U[i, j+4, k, 1]; u12 = U[i, j+3, k, 1]; u13 = U[i, j+2, k, 1]; u14 = U[i, j+1, k, 1]; u15 = U[i, j, k, 1]; u16 = U[i, j-1, k, 1]; u17 = U[i, j-2, k, 1]
        u21 = U[i, j+4, k, 2]; u22 = U[i, j+3, k, 2]; u23 = U[i, j+2, k, 2]; u24 = U[i, j+1, k, 2]; u25 = U[i, j, k, 2]; u26 = U[i, j-1, k, 2]; u27 = U[i, j-2, k, 2]
        u31 = U[i, j+4, k, 3]; u32 = U[i, j+3, k, 3]; u33 = U[i, j+2, k, 3]; u34 = U[i, j+1, k, 3]; u35 = U[i, j, k, 3]; u36 = U[i, j-1, k, 3]; u37 = U[i, j-2, k, 3]
        u41 = U[i, j+4, k, 4]; u42 = U[i, j+3, k, 4]; u43 = U[i, j+2, k, 4]; u44 = U[i, j+1, k, 4]; u45 = U[i, j, k, 4]; u46 = U[i, j-1, k, 4]; u47 = U[i, j-2, k, 4]
        u51 = U[i, j+4, k, 5]; u52 = U[i, j+3, k, 5]; u53 = U[i, j+2, k, 5]; u54 = U[i, j+1, k, 5]; u55 = U[i, j, k, 5]; u56 = U[i, j-1, k, 5]; u57 = U[i, j-2, k, 5]
    end
    UR = SMatrix{5,7,Float32}(
        u11, u21, u31, u41, u51,
        u12, u22, u32, u42, u52,
        u13, u23, u33, u43, u53,
        u14, u24, u34, u44, u54,
        u15, u25, u35, u45, u55,
        u16, u26, u36, u46, u56,
        u17, u27, u37, u47, u57
    )
    UL_interp = MVector{5,Float32}(undef)
    UR_interp = MVector{5,Float32}(undef)

    # Jameson sensor (J-direction)
    @inbounds ϕx = max(ϕ[i, j-2, k], 
                       ϕ[i, j-1, k], 
                       ϕ[i, j  , k], 
                       ϕ[i, j+1, k], 
                       ϕ[i, j+2, k], 
                       ϕ[i, j+3, k])
    
    if ϕx < hybrid_ϕ1   # smooth region
        c_vec = SVector{7,Float32}(Linear[1], Linear[2], Linear[3], Linear[4], Linear[5], Linear[6], Linear[7])
        UL_interp .= UL * c_vec
        UR_interp .= UR * c_vec
    else # discontinuous region
        
        # ==========================================
        # 2. Geometry at j+1/2
        # ==========================================

        # Tangent vectors construction
        @inbounds if abs(nz) <= abs(ny)
            @inbounds den::Float32 = sqrt(nx*nx + ny*ny + 1e-12f0)
            @inbounds lx::Float32 = -ny / den
            @inbounds ly::Float32 =  nx / den
            @inbounds lz::Float32 =  0.0f0
        else
            @inbounds den::Float32 = sqrt(nx*nx + nz*nz + 1e-12f0)
            @inbounds lx::Float32 = -nz / den
            @inbounds ly::Float32 =  0.0f0
            @inbounds lz::Float32 =  nx / den
        end
        @inbounds mx::Float32 = ny * lz - nz * ly
        @inbounds my::Float32 = nz * lx - nx * lz
        @inbounds mz::Float32 = nx * ly - ny * lx

        # ==========================================
        # 3. Roe Averages at j+1/2
        # ==========================================
        @inbounds ρ = sqrt(Q[i, j, k, 1] * Q[i, j+1, k, 1])
        @inbounds u = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 2] + sqrt(Q[i, j+1, k, 1]) * Q[i, j+1, k, 2]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        @inbounds v = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 3] + sqrt(Q[i, j+1, k, 1]) * Q[i, j+1, k, 3]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        @inbounds w = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 4] + sqrt(Q[i, j+1, k, 1]) * Q[i, j+1, k, 4]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        @inbounds HL = γ/(γ-1f0)*Q[i, j, k, 5]/Q[i, j, k, 1] + 0.5f0*(Q[i, j, k, 2]^2 + Q[i, j, k, 3]^2 + Q[i, j, k, 4]^2)
        @inbounds HR = γ/(γ-1f0)*Q[i, j+1, k, 5]/Q[i, j+1, k, 1] + 0.5f0*(Q[i, j+1, k, 2]^2 + Q[i, j+1, k, 3]^2 + Q[i, j+1, k, 4]^2)
        @inbounds H = (sqrt(Q[i, j, k, 1]) * HL + sqrt(Q[i, j+1, k, 1]) * HR) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j+1, k, 1]))
        @inbounds v2 = 0.5f0*(u^2 + v^2 + w^2)
        @inbounds c = sqrt((γ-1f0)*(H - v2))
        @inbounds un = u*nx + v*ny + w*nz
        @inbounds ul = u*lx + v*ly + w*lz
        @inbounds um = u*mx + v*my + w*mz
        @inbounds K = γ-1f0

        # Eigenmatrices construction (identical structure to I-direction)
        @inbounds begin
            invc = 1f0/c; invc2 = invc*invc
            Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
            Kv2 = K*v2*invc2; Kc2 = K*invc2
            un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
            half = 0.5f0; mhalf = -0.5f0
        end
        # ... (Ln11 to Rn55 definitions are identical since they depend on nx,ny,nz,u,v,w,c) ...
        @inbounds Ln11 = half*(Kv2 + un_invc);  Ln12 = mhalf*(Ku + nx_invc); Ln13 = mhalf*(Kv + ny_invc); Ln14 = mhalf*(Kw + nz_invc); Ln15 = half*Kc2
        @inbounds Ln21 = 1f0 - Kv2;             Ln22 = Ku;                   Ln23 = Kv;                   Ln24 = Kw;                   Ln25 = -Kc2
        @inbounds Ln31 = half*(Kv2 - un_invc);  Ln32 = mhalf*(Ku - nx_invc); Ln33 = mhalf*(Kv - ny_invc); Ln34 = mhalf*(Kw - nz_invc); Ln35 = half*Kc2
        @inbounds Ln41 = -ul;                   Ln42 = lx;                   Ln43 = ly;                   Ln44 = lz;                   Ln45 = 0f0
        @inbounds Ln51 = -um;                   Ln52 = mx;                   Ln53 = my;                   Ln54 = mz;                   Ln55 = 0f0

        @inbounds Rn11 = 1f0;            Rn12 = 1f0;      Rn13 = 1f0;            Rn14 = 0f0;      Rn15 = 0f0
        @inbounds Rn21 = u - nx*c;     Rn22 = u;        Rn23 = u + nx*c;     Rn24 = lx;     Rn25 = mx
        @inbounds Rn31 = v - ny*c;     Rn32 = v;        Rn33 = v + ny*c;     Rn34 = ly;     Rn35 = my
        @inbounds Rn41 = w - nz*c;     Rn42 = w;        Rn43 = w + nz*c;     Rn44 = lz;     Rn45 = mz
        @inbounds Rn51 = H - un*c;     Rn52 = v2;       Rn53 = H + un*c;     Rn54 = ul;     Rn55 = um

        L = SMatrix{5,5,Float32}(Ln11, Ln21, Ln31, Ln41, Ln51, Ln12, Ln22, Ln32, Ln42, Ln52, Ln13, Ln23, Ln33, Ln43, Ln53, Ln14, Ln24, Ln34, Ln44, Ln54, Ln15, Ln25, Ln35, Ln45, Ln55)
        R = SMatrix{5,5,Float32}(Rn11, Rn21, Rn31, Rn41, Rn51, Rn12, Rn22, Rn32, Rn42, Rn52, Rn13, Rn23, Rn33, Rn43, Rn53, Rn14, Rn24, Rn34, Rn44, Rn54, Rn15, Rn25, Rn35, Rn45, Rn55)

        # Constants
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0; WENOϵ1 = 1e-20; WENOϵ2 = 1f-16
        @inbounds ss = 2.0f0/(S[i, j+1, k] + S[i, j, k]) # Note: S indices changed

        # Characteristic Decomposition
        WL = L * UL; WR = L * UR
        WL_interp = MVector{5,Float32}(undef)
        WR_interp = MVector{5,Float32}(undef)

        if ϕx < hybrid_ϕ2   # WENO7
            for n = 1:NV
                @inbounds V1L = WL[n, 1]; V1R = WR[n, 1]
                @inbounds V2L = WL[n, 2]; V2R = WR[n, 2]
                @inbounds V3L = WL[n, 3]; V3R = WR[n, 3]
                @inbounds V4L = WL[n, 4]; V4R = WR[n, 4]
                @inbounds V5L = WL[n, 5]; V5R = WR[n, 5]
                @inbounds V6L = WL[n, 6]; V6R = WR[n, 6]
                @inbounds V7L = WL[n, 7]; V7R = WR[n, 7]

                q1L = -3V1L+13V2L-23V3L+25V4L ; q1R = -3V1R+13V2R-23V3R+25V4R
                q2L = V2L-5V3L+13V4L+3V5L     ; q2R = V2R-5V3R+13V4R+3V5R
                q3L = -V3L+7V4L+7V5L-V6L      ; q3R = -V3R+7V4R+7V5R-V6R
                q4L = 3V4L+13V5L-5V6L+V7L     ; q4R = 3V4R+13V5R-5V6R+V7R

                Is1L = V1L*(547V1L - 3882V2L + 4642V3L - 1854V4L) + V2L*(7043V2L -17246V3L + 7042V4L) + V3L*(11003V3L - 9402V4L) + V4L*(2107V4L)
                Is2L = V2L*(267V2L - 1642V3L + 1602V4L - 494V5L) + V3L*(2843V3L - 5966V4L + 1922V5L) + V4L*(3443V4L - 2522V5L) + V5L*(547V5L)
                Is3L = V3L*(547V3L - 2522V4L + 1922V5L - 494V6L) + V4L*(3443V4L - 5966V5L + 1602V6L) + V5L*(2843V5L - 1642V6L) + V6L*(267V6L)
                Is4L = V4L*(2107V4L - 9402V5L + 7042V6L - 1854V7L) + V5L*(11003V5L -17246V6L + 4642V7L) + V6L*(7043V6L - 3882V7L) + V7L*(547V7L)

                Is1R = V1R*(547V1R - 3882V2R + 4642V3R - 1854V4R) + V2R*(7043V2R -17246V3R + 7042V4R) + V3R*(11003V3R - 9402V4R) + V4R*(2107V4R)
                Is2R = V2R*(267V2R - 1642V3R + 1602V4R - 494V5R) + V3R*(2843V3R - 5966V4R + 1922V5R) + V4R*(3443V4R - 2522V5R) + V5R*(547V5R)
                Is3R = V3R*(547V3R - 2522V4R + 1922V5R - 494V6R) + V4R*(3443V4R - 5966V5R + 1602V6R) + V5R*(2843V5R - 1642V6R) + V6R*(267V6R)
                Is4R = V4R*(2107V4R - 9402V5R + 7042V6R - 1854V7R) + V5R*(11003V5R -17246V6R + 4642V7R) + V6R*(7043V6R - 3882V7R) + V7R*(547V7R)

                denom = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(denom*denom); denom = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(denom*denom)
                denom = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(denom*denom); denom = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(denom*denom)
                denom = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(denom*denom); denom = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(denom*denom)
                denom = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(denom*denom); denom = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(denom*denom)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                WL_interp[n] = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                WR_interp[n] = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1
            end
            UL_interp .= R * WL_interp; UR_interp .= R * WR_interp

        elseif ϕx < hybrid_ϕ3   # WENO5
            for n = 1:NV
                @inbounds V1L = WL[n, 2]; V1R = WR[n, 2]
                @inbounds V2L = WL[n, 3]; V2R = WR[n, 3]
                @inbounds V3L = WL[n, 4]; V3R = WR[n, 4]
                @inbounds V4L = WL[n, 5]; V4R = WR[n, 5]
                @inbounds V5L = WL[n, 6]; V5R = WR[n, 6]

                s11L = 13*(V1L-2*V2L+V3L)^2 + 3*(V1L-4*V2L+3*V3L)^2; s11R = 13*(V1R-2*V2R+V3R)^2 + 3*(V1R-4*V2R+3*V3R)^2 
                s22L = 13*(V2L-2*V3L+V4L)^2 + 3*(V2L-V4L)^2        ; s22R = 13*(V2R-2*V3R+V4R)^2 + 3*(V2R-V4R)^2        
                s33L = 13*(V3L-2*V4L+V5L)^2 + 3*(3*V3L-4*V4L+V5L)^2; s33R = 13*(V3R-2*V4R+V5R)^2 + 3*(3*V3R-4*V4R+V5R)^2 

                denom = WENOϵ2 + s11L*ss; s11L = 1.0f0/(denom*denom); denom = WENOϵ2 + s11R*ss; s11R = 1.0f0/(denom*denom)
                denom = WENOϵ2 + s22L*ss; s22L = 6.0f0/(denom*denom); denom = WENOϵ2 + s22R*ss; s22R = 6.0f0/(denom*denom)
                denom = WENOϵ2 + s33L*ss; s33L = 3.0f0/(denom*denom); denom = WENOϵ2 + s33R*ss; s33R = 3.0f0/(denom*denom)
                
                invsumL = 1.0f0/(s11L+s22L+s33L); invsumR = 1.0f0/(s11R+s22R+s33R)
                v1L = 2*V1L-7*V2L+11*V3L; v1R = 2*V1R-7*V2R+11*V3R
                v2L = -V2L+5*V3L+2*V4L  ; v2R = -V2R+5*V3R+2*V4R
                v3L = 2*V3L+5*V4L-V5L   ; v3R = 2*V3R+5*V4R-V5R
                WL_interp[n] = invsumL*(s11L*v1L+s22L*v2L+s33L*v3L) * tmp2
                WR_interp[n] = invsumR*(s11R*v1R+s22R*v2R+s33R*v3R) * tmp2
            end
            UL_interp .= R * WL_interp; UR_interp .= R * WR_interp
        else    # Minmod
            for n = 1:NV
                @inbounds WL_interp[n] = WL[n, 4] + 0.5f0*minmod(WL[n, 4] - WL[n, 3], WL[n, 5] - WL[n, 4])
                @inbounds WR_interp[n] = WR[n, 4] - 0.5f0*minmod(WR[n, 3] - WR[n, 4], WR[n, 4] - WR[n, 5])
            end 
            UL_interp .= R * WL_interp; UR_interp .= R * WR_interp
        end
    end
    Fy[i, j, k, :] = HLLC_Flux(UL_interp, UR_interp, nx, ny, nz)
    
end

function Eigen_reconstruct_k(Q, U, ϕ, S, Fz, dζdx, dζdy, dζdz)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    # 边界检查：注意这里主要防止 k 方向越界
    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG
        return
    end
    
    @inbounds nx::Float32 = (dζdx[i, j, k] + dζdx[i, j, k+1]) * 0.5f0
    @inbounds ny::Float32 = (dζdy[i, j, k] + dζdy[i, j, k+1]) * 0.5f0
    @inbounds nz::Float32 = (dζdz[i, j, k] + dζdz[i, j, k+1]) * 0.5f0
    @inbounds inv_len = 1.0f0 / sqrt(nx*nx + ny*ny + nz*nz + 1e-12f0)
    @inbounds nx *= inv_len
    @inbounds ny *= inv_len
    @inbounds nz *= inv_len

    # ==========================================
    # 1. Load Stencil (K-direction shift)
    # ==========================================
    # left state reconstruction at k+1/2
    @inbounds begin
        u11 = U[i, j, k-3, 1]; u12 = U[i, j, k-2, 1]; u13 = U[i, j, k-1, 1]; u14 = U[i, j, k, 1]; u15 = U[i, j, k+1, 1]; u16 = U[i, j, k+2, 1]; u17 = U[i, j, k+3, 1]
        u21 = U[i, j, k-3, 2]; u22 = U[i, j, k-2, 2]; u23 = U[i, j, k-1, 2]; u24 = U[i, j, k, 2]; u25 = U[i, j, k+1, 2]; u26 = U[i, j, k+2, 2]; u27 = U[i, j, k+3, 2]
        u31 = U[i, j, k-3, 3]; u32 = U[i, j, k-2, 3]; u33 = U[i, j, k-1, 3]; u34 = U[i, j, k, 3]; u35 = U[i, j, k+1, 3]; u36 = U[i, j, k+2, 3]; u37 = U[i, j, k+3, 3]
        u41 = U[i, j, k-3, 4]; u42 = U[i, j, k-2, 4]; u43 = U[i, j, k-1, 4]; u44 = U[i, j, k, 4]; u45 = U[i, j, k+1, 4]; u46 = U[i, j, k+2, 4]; u47 = U[i, j, k+3, 4]
        u51 = U[i, j, k-3, 5]; u52 = U[i, j, k-2, 5]; u53 = U[i, j, k-1, 5]; u54 = U[i, j, k, 5]; u55 = U[i, j, k+1, 5]; u56 = U[i, j, k+2, 5]; u57 = U[i, j, k+3, 5]
    end
    UL = SMatrix{5,7,Float32}(
        u11, u21, u31, u41, u51,
        u12, u22, u32, u42, u52,
        u13, u23, u33, u43, u53,
        u14, u24, u34, u44, u54,
        u15, u25, u35, u45, u55,
        u16, u26, u36, u46, u56,
        u17, u27, u37, u47, u57
    )

    # right state reconstruction at k+1/2 (Mirrored in K)
    @inbounds begin
        u11 = U[i, j, k+4, 1]; u12 = U[i, j, k+3, 1]; u13 = U[i, j, k+2, 1]; u14 = U[i, j, k+1, 1]; u15 = U[i, j, k, 1]; u16 = U[i, j, k-1, 1]; u17 = U[i, j, k-2, 1]
        u21 = U[i, j, k+4, 2]; u22 = U[i, j, k+3, 2]; u23 = U[i, j, k+2, 2]; u24 = U[i, j, k+1, 2]; u25 = U[i, j, k, 2]; u26 = U[i, j, k-1, 2]; u27 = U[i, j, k-2, 2]
        u31 = U[i, j, k+4, 3]; u32 = U[i, j, k+3, 3]; u33 = U[i, j, k+2, 3]; u34 = U[i, j, k+1, 3]; u35 = U[i, j, k, 3]; u36 = U[i, j, k-1, 3]; u37 = U[i, j, k-2, 3]
        u41 = U[i, j, k+4, 4]; u42 = U[i, j, k+3, 4]; u43 = U[i, j, k+2, 4]; u44 = U[i, j, k+1, 4]; u45 = U[i, j, k, 4]; u46 = U[i, j, k-1, 4]; u47 = U[i, j, k-2, 4]
        u51 = U[i, j, k+4, 5]; u52 = U[i, j, k+3, 5]; u53 = U[i, j, k+2, 5]; u54 = U[i, j, k+1, 5]; u55 = U[i, j, k, 5]; u56 = U[i, j, k-1, 5]; u57 = U[i, j, k-2, 5]
    end
    UR = SMatrix{5,7,Float32}(
        u11, u21, u31, u41, u51,
        u12, u22, u32, u42, u52,
        u13, u23, u33, u43, u53,
        u14, u24, u34, u44, u54,
        u15, u25, u35, u45, u55,
        u16, u26, u36, u46, u56,
        u17, u27, u37, u47, u57
    )
    UL_interp = MVector{5,Float32}(undef)
    UR_interp = MVector{5,Float32}(undef)

    # Jameson sensor (K-direction)
    @inbounds ϕx = max(ϕ[i, j, k-2], 
                       ϕ[i, j, k-1], 
                       ϕ[i, j, k  ], 
                       ϕ[i, j, k+1], 
                       ϕ[i, j, k+2], 
                       ϕ[i, j, k+3])
    
    if ϕx < hybrid_ϕ1   # smooth region
        c_vec = SVector{7,Float32}(Linear[1], Linear[2], Linear[3], Linear[4], Linear[5], Linear[6], Linear[7])
        UL_interp .= UL * c_vec
        UR_interp .= UR * c_vec
    else # discontinuous region
        
        # ==========================================
        # 2. Geometry at k+1/2
        # ==========================================

        # Tangent vectors construction
        @inbounds if abs(nz) <= abs(ny)
            @inbounds den::Float32 = sqrt(nx*nx + ny*ny + 1e-12f0)
            @inbounds lx::Float32 = -ny / den
            @inbounds ly::Float32 =  nx / den
            @inbounds lz::Float32 =  0.0f0
        else
            @inbounds den::Float32 = sqrt(nx*nx + nz*nz + 1e-12f0)
            @inbounds lx::Float32 = -nz / den
            @inbounds ly::Float32 =  0.0f0
            @inbounds lz::Float32 =  nx / den
        end
        @inbounds mx::Float32 = ny * lz - nz * ly
        @inbounds my::Float32 = nz * lx - nx * lz
        @inbounds mz::Float32 = nx * ly - ny * lx

        # ==========================================
        # 3. Roe Averages at k+1/2
        # ==========================================
        @inbounds ρ = sqrt(Q[i, j, k, 1] * Q[i, j, k+1, 1])
        @inbounds u = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 2] + sqrt(Q[i, j, k+1, 1]) * Q[i, j, k+1, 2]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        @inbounds v = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 3] + sqrt(Q[i, j, k+1, 1]) * Q[i, j, k+1, 3]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        @inbounds w = (sqrt(Q[i, j, k, 1]) * Q[i, j, k, 4] + sqrt(Q[i, j, k+1, 1]) * Q[i, j, k+1, 4]) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        @inbounds HL = γ/(γ-1f0)*Q[i, j, k, 5]/Q[i, j, k, 1] + 0.5f0*(Q[i, j, k, 2]^2 + Q[i, j, k, 3]^2 + Q[i, j, k, 4]^2)
        @inbounds HR = γ/(γ-1f0)*Q[i, j, k+1, 5]/Q[i, j, k+1, 1] + 0.5f0*(Q[i, j, k+1, 2]^2 + Q[i, j, k+1, 3]^2 + Q[i, j, k+1, 4]^2)
        @inbounds H = (sqrt(Q[i, j, k, 1]) * HL + sqrt(Q[i, j, k+1, 1]) * HR) / (sqrt(Q[i, j, k, 1]) + sqrt(Q[i, j, k+1, 1]))
        @inbounds v2 = 0.5f0*(u^2 + v^2 + w^2)
        @inbounds c = sqrt((γ-1f0)*(H - v2))
        @inbounds un = u*nx + v*ny + w*nz
        @inbounds ul = u*lx + v*ly + w*lz
        @inbounds um = u*mx + v*my + w*mz
        @inbounds K = γ-1f0

        # Eigenmatrices construction
        @inbounds begin
            invc = 1f0/c; invc2 = invc*invc
            Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
            Kv2 = K*v2*invc2; Kc2 = K*invc2
            un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
            half = 0.5f0; mhalf = -0.5f0
        end
        # ... (Ln11 to Rn55 definitions are identical) ...
        @inbounds Ln11 = half*(Kv2 + un_invc);  Ln12 = mhalf*(Ku + nx_invc); Ln13 = mhalf*(Kv + ny_invc); Ln14 = mhalf*(Kw + nz_invc); Ln15 = half*Kc2
        @inbounds Ln21 = 1f0 - Kv2;             Ln22 = Ku;                   Ln23 = Kv;                   Ln24 = Kw;                   Ln25 = -Kc2
        @inbounds Ln31 = half*(Kv2 - un_invc);  Ln32 = mhalf*(Ku - nx_invc); Ln33 = mhalf*(Kv - ny_invc); Ln34 = mhalf*(Kw - nz_invc); Ln35 = half*Kc2
        @inbounds Ln41 = -ul;                   Ln42 = lx;                   Ln43 = ly;                   Ln44 = lz;                   Ln45 = 0f0
        @inbounds Ln51 = -um;                   Ln52 = mx;                   Ln53 = my;                   Ln54 = mz;                   Ln55 = 0f0

        @inbounds Rn11 = 1f0;            Rn12 = 1f0;      Rn13 = 1f0;            Rn14 = 0f0;      Rn15 = 0f0
        @inbounds Rn21 = u - nx*c;     Rn22 = u;        Rn23 = u + nx*c;     Rn24 = lx;     Rn25 = mx
        @inbounds Rn31 = v - ny*c;     Rn32 = v;        Rn33 = v + ny*c;     Rn34 = ly;     Rn35 = my
        @inbounds Rn41 = w - nz*c;     Rn42 = w;        Rn43 = w + nz*c;     Rn44 = lz;     Rn45 = mz
        @inbounds Rn51 = H - un*c;     Rn52 = v2;       Rn53 = H + un*c;     Rn54 = ul;     Rn55 = um

        L = SMatrix{5,5,Float32}(Ln11, Ln21, Ln31, Ln41, Ln51, Ln12, Ln22, Ln32, Ln42, Ln52, Ln13, Ln23, Ln33, Ln43, Ln53, Ln14, Ln24, Ln34, Ln44, Ln54, Ln15, Ln25, Ln35, Ln45, Ln55)
        R = SMatrix{5,5,Float32}(Rn11, Rn21, Rn31, Rn41, Rn51, Rn12, Rn22, Rn32, Rn42, Rn52, Rn13, Rn23, Rn33, Rn43, Rn53, Rn14, Rn24, Rn34, Rn44, Rn54, Rn15, Rn25, Rn35, Rn45, Rn55)

        # Constants
        tmp1 = 1.0f0/12.0f0; tmp2 = 1.0f0/6.0f0; WENOϵ1 = 1e-20; WENOϵ2 = 1f-16
        @inbounds ss = 2.0f0/(S[i, j, k+1] + S[i, j, k]) # Note: S indices changed

        # Characteristic Decomposition
        WL = L * UL; WR = L * UR
        WL_interp = MVector{5,Float32}(undef)
        WR_interp = MVector{5,Float32}(undef)

        if ϕx < hybrid_ϕ2   # WENO7
            for n = 1:NV
                @inbounds V1L = WL[n, 1]; V1R = WR[n, 1]
                @inbounds V2L = WL[n, 2]; V2R = WR[n, 2]
                @inbounds V3L = WL[n, 3]; V3R = WR[n, 3]
                @inbounds V4L = WL[n, 4]; V4R = WR[n, 4]
                @inbounds V5L = WL[n, 5]; V5R = WR[n, 5]
                @inbounds V6L = WL[n, 6]; V6R = WR[n, 6]
                @inbounds V7L = WL[n, 7]; V7R = WR[n, 7]

                q1L = -3V1L+13V2L-23V3L+25V4L ; q1R = -3V1R+13V2R-23V3R+25V4R
                q2L = V2L-5V3L+13V4L+3V5L     ; q2R = V2R-5V3R+13V4R+3V5R
                q3L = -V3L+7V4L+7V5L-V6L      ; q3R = -V3R+7V4R+7V5R-V6R
                q4L = 3V4L+13V5L-5V6L+V7L     ; q4R = 3V4R+13V5R-5V6R+V7R

                Is1L = V1L*(547V1L - 3882V2L + 4642V3L - 1854V4L) + V2L*(7043V2L -17246V3L + 7042V4L) + V3L*(11003V3L - 9402V4L) + V4L*(2107V4L)
                Is2L = V2L*(267V2L - 1642V3L + 1602V4L - 494V5L) + V3L*(2843V3L - 5966V4L + 1922V5L) + V4L*(3443V4L - 2522V5L) + V5L*(547V5L)
                Is3L = V3L*(547V3L - 2522V4L + 1922V5L - 494V6L) + V4L*(3443V4L - 5966V5L + 1602V6L) + V5L*(2843V5L - 1642V6L) + V6L*(267V6L)
                Is4L = V4L*(2107V4L - 9402V5L + 7042V6L - 1854V7L) + V5L*(11003V5L -17246V6L + 4642V7L) + V6L*(7043V6L - 3882V7L) + V7L*(547V7L)

                Is1R = V1R*(547V1R - 3882V2R + 4642V3R - 1854V4R) + V2R*(7043V2R -17246V3R + 7042V4R) + V3R*(11003V3R - 9402V4R) + V4R*(2107V4R)
                Is2R = V2R*(267V2R - 1642V3R + 1602V4R - 494V5R) + V3R*(2843V3R - 5966V4R + 1922V5R) + V4R*(3443V4R - 2522V5R) + V5R*(547V5R)
                Is3R = V3R*(547V3R - 2522V4R + 1922V5R - 494V6R) + V4R*(3443V4R - 5966V5R + 1602V6R) + V5R*(2843V5R - 1642V6R) + V6R*(267V6R)
                Is4R = V4R*(2107V4R - 9402V5R + 7042V6R - 1854V7R) + V5R*(11003V5R -17246V6R + 4642V7R) + V6R*(7043V6R - 3882V7R) + V7R*(547V7R)

                denom = WENOϵ1 + Is1L*ss; α1L = 1.0f0/(denom*denom); denom = WENOϵ1 + Is1R*ss; α1R = 1.0f0/(denom*denom)
                denom = WENOϵ1 + Is2L*ss; α2L = 12.0f0/(denom*denom); denom = WENOϵ1 + Is2R*ss; α2R = 12.0f0/(denom*denom)
                denom = WENOϵ1 + Is3L*ss; α3L = 18.0f0/(denom*denom); denom = WENOϵ1 + Is3R*ss; α3R = 18.0f0/(denom*denom)
                denom = WENOϵ1 + Is4L*ss; α4L = 4.0f0/(denom*denom); denom = WENOϵ1 + Is4R*ss; α4R = 4.0f0/(denom*denom)

                invsumL = 1.0f0/(α1L+α2L+α3L+α4L); invsumR = 1.0f0/(α1R+α2R+α3R+α4R)
                WL_interp[n] = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                WR_interp[n] = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1
            end
            UL_interp .= R * WL_interp; UR_interp .= R * WR_interp

        elseif ϕx < hybrid_ϕ3   # WENO5
            for n = 1:NV
                @inbounds V1L = WL[n, 2]; V1R = WR[n, 2]
                @inbounds V2L = WL[n, 3]; V2R = WR[n, 3]
                @inbounds V3L = WL[n, 4]; V3R = WR[n, 4]
                @inbounds V4L = WL[n, 5]; V4R = WR[n, 5]
                @inbounds V5L = WL[n, 6]; V5R = WR[n, 6]

                s11L = 13*(V1L-2*V2L+V3L)^2 + 3*(V1L-4*V2L+3*V3L)^2; s11R = 13*(V1R-2*V2R+V3R)^2 + 3*(V1R-4*V2R+3*V3R)^2 
                s22L = 13*(V2L-2*V3L+V4L)^2 + 3*(V2L-V4L)^2        ; s22R = 13*(V2R-2*V3R+V4R)^2 + 3*(V2R-V4R)^2        
                s33L = 13*(V3L-2*V4L+V5L)^2 + 3*(3*V3L-4*V4L+V5L)^2; s33R = 13*(V3R-2*V4R+V5R)^2 + 3*(3*V3R-4*V4R+V5R)^2 

                denom = WENOϵ2 + s11L*ss; s11L = 1.0f0/(denom*denom); denom = WENOϵ2 + s11R*ss; s11R = 1.0f0/(denom*denom)
                denom = WENOϵ2 + s22L*ss; s22L = 6.0f0/(denom*denom); denom = WENOϵ2 + s22R*ss; s22R = 6.0f0/(denom*denom)
                denom = WENOϵ2 + s33L*ss; s33L = 3.0f0/(denom*denom); denom = WENOϵ2 + s33R*ss; s33R = 3.0f0/(denom*denom)
                
                invsumL = 1.0f0/(s11L+s22L+s33L); invsumR = 1.0f0/(s11R+s22R+s33R)
                v1L = 2*V1L-7*V2L+11*V3L; v1R = 2*V1R-7*V2R+11*V3R
                v2L = -V2L+5*V3L+2*V4L  ; v2R = -V2R+5*V3R+2*V4R
                v3L = 2*V3L+5*V4L-V5L   ; v3R = 2*V3R+5*V4R-V5R
                WL_interp[n] = invsumL*(s11L*v1L+s22L*v2L+s33L*v3L) * tmp2
                WR_interp[n] = invsumR*(s11R*v1R+s22R*v2R+s33R*v3R) * tmp2
            end
            UL_interp .= R * WL_interp; UR_interp .= R * WR_interp
        else    # Minmod
            for n = 1:NV
                @inbounds WL_interp[n] = WL[n, 4] + 0.5f0*minmod(WL[n, 4] - WL[n, 3], WL[n, 5] - WL[n, 4])
                @inbounds WR_interp[n] = WR[n, 4] - 0.5f0*minmod(WR[n, 3] - WR[n, 4], WR[n, 4] - WR[n, 5])
            end 
            UL_interp .= R * WL_interp; UR_interp .= R * WR_interp
        end
    end
    Fz[i, j, k, :] = HLLC_Flux(UL_interp, UR_interp, nx, ny, nz)
    
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
             (ρL * (SL - qL) - ρR * (SR - qR) + 1e-12f0) # 加极小量防除零

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

function Riemann_Solver(Q, U, ϕ, S, Fx, Fy, Fz, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz)
    if eigen_reconstruction
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Eigen_reconstruct_i(Q, U, ϕ, S, Fx, dξdx, dξdy, dξdz)
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Eigen_reconstruct_j(Q, U, ϕ, S, Fy, dηdx, dηdy, dηdz)
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock Eigen_reconstruct_k(Q, U, ϕ, S, Fz, dζdx, dζdy, dζdz)
    end
    
    

end