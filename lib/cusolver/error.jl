export CUSOLVERError

struct CUSOLVERError <: Exception
    code::cusolverStatus_t
end

Base.convert(::Type{cusolverStatus_t}, err::CUSOLVERError) = err.code

Base.showerror(io::IO, err::CUSOLVERError) =
    print(io, "CUSOLVERError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUSOLVERError) = string(err.code)

## COV_EXCL_START
function description(err)
    if err.code == CUSOLVER_STATUS_SUCCESS
        "the operation completed successfully"
    elseif err.code == CUSOLVER_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUSOLVER_STATUS_ALLOC_FAILED
        "the resource allocation failed"
    elseif err.code == CUSOLVER_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUSOLVER_STATUS_ARCH_MISMATCH
        "an absent device architectural feature is required"
    elseif err.code == CUSOLVER_STATUS_EXECUTION_FAILED
        "the GPU program failed to execute"
    elseif err.code == CUSOLVER_STATUS_INTERNAL_ERROR
        "an internal operation failed"
    elseif err.code == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        "the matrix type is not supported."
    elseif err.code == CUSOLVER_STATUS_NOT_SUPPORTED
        "the parameter combination is not supported."
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP
