import KernelAbstractions
import KernelAbstractions as KA

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

ka_skip_tests = Set{String}(["CPU synchronization", "fallback test: callable types"])
Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDA, CuArray, CuDeviceArray; skip_tests=ka_skip_tests)
for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
    Testsuite.unittest_testsuite(()->CUDABackend(PreferBlocks, AlwaysInline), "CUDA", CUDA, CuDeviceArray;
                                 skip_tests=ka_skip_tests)
end

@testset "KA.functional" begin
    @test KA.functional(CUDABackend()) == CUDA.functional()
end
