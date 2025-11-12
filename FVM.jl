function variable_reconstruction_x(U, ULx, URx)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp || j > Nyp || k > Nzp
        return
    end

    
    
    return 
end

