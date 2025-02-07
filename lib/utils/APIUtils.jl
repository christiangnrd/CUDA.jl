module APIUtils

using ..CUDA

using LLVM
using LLVM.Interop

# helpers that facilitate working with CUDA APIs
using GPUUtils: @checked, @debug_ccall
export @checked, @debug_ccall, @gcsafe_ccall

include("enum.jl")
include("threading.jl")
include("cache.jl")
include("memoization.jl")
include("struct_size.jl")

end
