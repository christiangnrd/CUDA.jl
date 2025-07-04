using CUDA.CUSPARSE
using SparseArrays
using LinearAlgebra

if CUSPARSE.version() >= v"11.4.1"
    @testset "generic mv!" for T in [Float32, Float64]
        m = 10
        A = sprand(T, m, m, 0.1)
        x = rand(Complex{T}, m)
        y = similar(x)
        dx = adapt(CuArray, x)
        dy = adapt(CuArray, y)

        dA = adapt(CuArray, A)
        mv!('N', one(T), dA, dx, zero(T), dy, 'O')
        @test Array(dy) ≈ A * x

        dA = CuSparseMatrixCSR(dA)
        mv!('N', one(T), dA, dx, zero(T), dy, 'O')
        @test Array(dy) ≈ A * x
        
        A_bad = sprand(T, m+1, m, 0.1)
        dA_bad = adapt(CuArray, A_bad)
        @test_throws DimensionMismatch("Y must have length $(m+1), but has length $m") mv!('N', one(T), dA_bad, dx, zero(T), dy, 'O')

        A_bad = sprand(T, m, m+1, 0.1)
        dA_bad = adapt(CuArray, A_bad)
        @test_throws DimensionMismatch("X must have length $(m+1), but has length $m") mv!('N', one(T), dA_bad, dx, zero(T), dy, 'O')
    end
end

if CUSPARSE.version() >= v"11.4.1" # lower CUDA version doesn't support these algorithms

    SPMV_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMV_CSR_ALG1,
                                            CUSPARSE.CUSPARSE_SPMV_CSR_ALG2])

    SPMM_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG1,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG2,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG3])

    if CUSPARSE.version() >= v"11.7.2"
        SPMV_ALGOS[CuSparseMatrixCOO] = [CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT,
                                         CUSPARSE.CUSPARSE_SPMV_COO_ALG1]

        SPMM_ALGOS[CuSparseMatrixCOO] = [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                         CUSPARSE.CUSPARSE_SPMM_COO_ALG1,
                                         CUSPARSE.CUSPARSE_SPMM_COO_ALG3,
                                         CUSPARSE.CUSPARSE_SPMM_COO_ALG4]
    end

    if CUSPARSE.version() >= v"11.7.4"
        push!(SPMM_ALGOS[CuSparseMatrixCOO], CUSPARSE.CUSPARSE_SPMM_COO_ALG2)
    end

    if CUSPARSE.version() >= v"12.1.3"
        push!(SPMV_ALGOS[CuSparseMatrixCOO], CUSPARSE.CUSPARSE_SPMV_COO_ALG2)
    end

    if CUSPARSE.version() >= v"12.5.1"
        SPMM_ALGOS[CuSparseMatrixBSR] = [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                         CUSPARSE.CUSPARSE_SPMM_BSR_ALG1]
    end

    for SparseMatrixType in keys(SPMV_ALGOS)
        @testset "$SparseMatrixType -- mv! algo=$algo" for algo in SPMV_ALGOS[SparseMatrixType]
            @testset "mv! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                    A = sprand(T, 20, 10, 0.1)
                    B = transa == 'N' ? rand(T, 10) : rand(T, 20)
                    C = transa == 'N' ? rand(T, 20) : rand(T, 10)
                    dA = SparseMatrixType(A)
                    dB = CuArray(B)
                    dC = CuArray(C)

                    alpha = rand(T)
                    beta = rand(T)
                    mv!(transa, alpha, dA, dB, beta, dC, 'O', algo)
                    @test alpha * opa(A) * B + beta * C ≈ collect(dC)
                end
            end
        end
    end

    for SparseMatrixType in keys(SPMM_ALGOS)
        @testset "$SparseMatrixType * CuMatrix -- mm! algo=$algo" for algo in SPMM_ALGOS[SparseMatrixType]
            @testset "mm! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    @testset "transb = $transb" for (transb, opb) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                        algo == CUSPARSE.CUSPARSE_SPMM_CSR_ALG3 && (transa != 'N' || transb != 'N') && continue
                        (SparseMatrixType == CuSparseMatrixBSR) && (transa != 'N') && continue
                        A = sprand(T, 10, 10, 0.1)
                        B = transb == 'N' ? rand(T, 10, 2) : rand(T, 2, 10)
                        C = rand(T, 10, 2)
                        dA = SparseMatrixType == CuSparseMatrixBSR ? SparseMatrixType(A,1) : SparseMatrixType(A)
                        dB = CuArray(B)
                        dC = CuArray(C)

                        alpha = rand(T)
                        beta = rand(T)
                        mm!(transa, transb, alpha, dA, dB, beta, dC, 'O', algo)
                        @test alpha * opa(A) * opb(B) + beta * C ≈ collect(dC)
                    end
                end
            end
        end
    end
end

if CUSPARSE.version() >= v"11.7.4"

    # Algorithms for CSC and CSR matrices are swapped
    SPMM_ALGOS = Dict(CuSparseMatrixBSR => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT],
                      CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG1,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG2,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG3],
                      CuSparseMatrixCOO => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMM_COO_ALG1,
                                            CUSPARSE.CUSPARSE_SPMM_COO_ALG2,
                                            CUSPARSE.CUSPARSE_SPMM_COO_ALG3,
                                            CUSPARSE.CUSPARSE_SPMM_COO_ALG4])

    for SparseMatrixType in keys(SPMM_ALGOS)
        (SparseMatrixType == CuSparseMatrixBSR) && continue
        @testset "CuMatrix * $SparseMatrixType -- mm! algo=$algo" for algo in SPMM_ALGOS[SparseMatrixType]
            @testset "$T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    @testset "transb = $transb" for (transb, opb) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSR && T <: Complex && transb == 'C' && continue
                        algo == CUSPARSE.CUSPARSE_SPMM_CSR_ALG3 && (transa != 'N' || transb != 'N') && continue
                        A = rand(T, 10, 10)
                        B = transb == 'N' ? sprand(T, 10, 5, 0.5) : sprand(T, 5, 10, 0.5)
                        C = rand(T, 10, 5)
                        dA = CuArray(A)
                        dB = SparseMatrixType(B)
                        if SparseMatrixType == CuSparseMatrixCOO
                            dB = sort_coo(dB, 'C')
                        end
                        dC = CuArray(C)

                        alpha = rand(T)
                        beta = rand(T)
                        mm!(transa, transb, alpha, dA, dB, beta, dC, 'O', algo)
                        @test alpha * opa(A) * opb(B) + beta * C ≈ collect(dC)

                        # add tests for very small matrices (see #2296)
                        # skip conjugate transpose - causes errors with 1x1 matrices
                        # CUSPARSE_SPMM_CSR_ALG3 also fails
                        (algo == CUSPARSE.CUSPARSE_SPMM_CSR_ALG3 || transa == 'C') && continue
                        A = rand(T, 1, 1)
                        B = sprand(T, 1, 1, 1.)
                        C = rand(T, 1, 1)
                        dA = CuArray(A)
                        dB = SparseMatrixType(B)
                        dC = CuArray(C)

                        alpha = rand(T)
                        beta = rand(T)
                        mm!(transa, transb, alpha, dA, dB, beta, dC, 'O', algo)
                        @test alpha * opa(A) * opb(B) + beta * C ≈ collect(dC)
                    end
                end
            end
        end
    end
end

if CUSPARSE.version() >= v"12.0"

    SPSV_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT],
                      CuSparseMatrixCOO => [CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT])

    SPSM_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT],
                      CuSparseMatrixCOO => [CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT])

    for SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO]
        @testset "$SparseMatrixType -- sv! algo=$algo" for algo in SPSV_ALGOS[SparseMatrixType]
            @testset "sv! $T" for T in [Float64, ComplexF64]
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                    @testset "uplo = $uplo" for uplo in ('L', 'U')
                        @testset "diag = $diag" for diag in ('U', 'N')
                            A = rand(T, 10, 10)
                            A = uplo == 'L' ? tril(A) : triu(A)
                            A = diag == 'U' ? A - Diagonal(A) + I : A
                            A = sparse(A)
                            dA = SparseMatrixType(A)
                            B = rand(T, 10)
                            C = rand(T, 10)
                            dB = CuArray(B)
                            dC = CuArray(C)
                            alpha = rand(T)
                            sv!(transa, uplo, diag, alpha, dA, dB, dC, 'O', algo)
                            @test opa(A) \ (alpha * B) ≈ collect(dC)
                        end
                    end
                end
            end
        end

        @testset "$SparseMatrixType -- sm! algo=$algo" for algo in SPSM_ALGOS[SparseMatrixType]
            @testset "sm! $T" for T in [Float64, ComplexF64]
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                    @testset "transb = $transb" for (transb, opb) in [('N', identity), ('T', transpose)]
                        @testset "uplo = $uplo" for uplo in ('L', 'U')
                            @testset "diag = $diag" for diag in ('U', 'N')
                                A = rand(T, 10, 10)
                                A = uplo == 'L' ? tril(A) : triu(A)
                                A = diag == 'U' ? A - Diagonal(A) + I : A
                                A = sparse(A)
                                dA = SparseMatrixType(A)
                                B = transb == 'N' ? rand(T, 10, 2) : rand(T, 2, 10)
                                C = rand(T, 10, 2)
                                dB = CuArray(B)
                                dC = CuArray(C)
                                alpha = rand(T)
                                sm!(transa, transb, uplo, diag, alpha, dA, dB, dC, 'O', algo)
                                @test opa(A) \ (alpha * opb(B)) ≈ collect(dC)
                            end
                        end
                    end
                end
            end
        end
    end
end # CUSPARSE.version >= 12.0

fmt = Dict(CuSparseMatrixCSC => :csc,
           CuSparseMatrixCSR => :csr,
           CuSparseMatrixCOO => :coo)

for SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO]
    @testset "$SparseMatrixType -- densetosparse algo=$algo" for algo in [CUSPARSE.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT]
        @testset "densetosparse $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
            A_sparse = sprand(T, 10, 20, 0.5)
            A_dense = Matrix{T}(A_sparse)
            dA_dense = CuMatrix{T}(A_dense)
            dA_sparse = CUSPARSE.densetosparse(dA_dense, fmt[SparseMatrixType], 'O', algo)
            @test A_sparse ≈ collect(dA_sparse)
            @test_throws ArgumentError("Format :bad not available, use :csc, :csr or :coo.")  CUSPARSE.densetosparse(dA_dense, :bad, 'O', algo)
        end
    end
    @testset "$SparseMatrixType -- sparsetodense algo=$algo" for algo in [CUSPARSE.CUSPARSE_SPARSETODENSE_ALG_DEFAULT]
        @testset "sparsetodense $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
            A_dense = rand(T, 10, 20)
            A_sparse = sparse(A_dense)
            dA_sparse = SparseMatrixType(A_sparse)
            dA_dense = CUSPARSE.sparsetodense(dA_sparse, 'O', algo)
            @test A_dense ≈ collect(dA_dense)
        end
    end
end

@testset "vv! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "transx = $transx" for (transx, opx) in [('N', identity), ('C', conj)]
        T <: Real && transx == 'C' && continue
        X = sprand(T, 20, 0.5)
        dX = CuSparseVector{T}(X)
        Y = rand(T, 20)
        dY = CuVector{T}(Y)
        result = vv!(transx, dX, dY, 'O')
        @test sum(opx(X[i]) * Y[i] for i=1:20) ≈ result
    end
end

@testset "gather! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    X = sprand(T, 20, 0.5)
    dX = CuSparseVector{T}(X)
    Y = rand(T, 20)
    dY = CuVector{T}(Y)
    CUSPARSE.gather!(dX, dY, 'O')
    Z = copy(X)
    for i = 1:nnz(X)
        Z[X.nzind[i]] = Y[X.nzind[i]]
    end
    @test Z ≈ sparse(collect(dX))
end

@testset "scatter! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    X = sprand(T, 20, 0.5)
    dX = CuSparseVector{T}(X)
    Y = rand(T, 20)
    dY = CuVector{T}(Y)
    CUSPARSE.scatter!(dY, dX, 'O')
    Z = copy(Y)
    for i = 1:nnz(X)
        Z[X.nzind[i]] = X.nzval[i]
    end
    @test Z ≈ collect(dY)
end

@testset "axpby! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    X = sprand(T, 20, 0.5)
    dX = CuSparseVector{T}(X)
    Y = rand(T, 20)
    dY = CuVector{T}(Y)
    alpha = rand(T)
    beta = rand(T)
    CUSPARSE.axpby!(alpha, dX, beta, dY, 'O')
    @test alpha * X + beta * Y ≈ collect(dY)
end

@testset "rot! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    X = sprand(T, 20, 0.5)
    dX = CuSparseVector{T}(X)
    Y = rand(T, 20)
    dY = CuVector{T}(Y)
    c = rand(T)
    s = rand(T)
    CUSPARSE.rot!(dX, dY, c, s, 'O')
    W = copy(X)
    Z = copy(Y)
    for i = 1:nnz(X)
        W[X.nzind[i]] =  c * X.nzval[i] + s * Y[X.nzind[i]]
        Z[X.nzind[i]] = -s * X.nzval[i] + c * Y[X.nzind[i]]
    end
    @test W ≈ collect(dX)
    @test Z ≈ collect(dY)
end

SPGEMM_ALGOS = Dict(CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPGEMM_DEFAULT],
                    CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPGEMM_DEFAULT])
if CUSPARSE.version() >= v"12.0"
    append!(SPGEMM_ALGOS[CuSparseMatrixCSR], (CUSPARSE.CUSPARSE_SPGEMM_ALG1,
                                              CUSPARSE.CUSPARSE_SPGEMM_ALG2,
                                              CUSPARSE.CUSPARSE_SPGEMM_ALG3))
    append!(SPGEMM_ALGOS[CuSparseMatrixCSC], (CUSPARSE.CUSPARSE_SPGEMM_ALG1,
                                              CUSPARSE.CUSPARSE_SPGEMM_ALG2,
                                              CUSPARSE.CUSPARSE_SPGEMM_ALG3))
end
# Algorithms CUSPARSE.CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC and
# CUSPARSE.CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC are dedicated to the cusparseSpGEMMreuse routine.

for SparseMatrixType in keys(SPGEMM_ALGOS)
    @testset "$SparseMatrixType -- gemm -- gemm! algo=$algo" for algo in SPGEMM_ALGOS[SparseMatrixType]
        @testset "gemm -- gemm! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
            @testset "transa = $transa" for (transa, opa) in [('N', identity)]
                @testset "transb = $transb" for (transb, opb) in [('N', identity)]
                    A = sprand(T,25,10,0.2)
                    B = sprand(T,10,35,0.3)
                    dA = SparseMatrixType(A)
                    dB = SparseMatrixType(B)
                    alpha = rand(T)
                    C = alpha * opa(A) * opb(B)
                    dC = gemm(transa, transb, alpha, dA, dB, 'O', algo)
                    @test C ≈ SparseMatrixCSC(dC)

                    beta = rand(T)
                    gamma = rand(T)
                    D = gamma * opa(A) * opa(B) + beta * C

                    dD = gemm(transa, transb, gamma, dA, dB, beta, dC, 'O', algo, same_pattern=true)
                    @test D ≈ SparseMatrixCSC(dD)

                    gemm!(transa, transb, gamma, dA, dB, beta, dC, 'O', algo)
                    @test D ≈ SparseMatrixCSC(dC)

                    E = sprand(T,25,35,0.1)
                    dE = SparseMatrixType(E)
                    F = alpha * opa(A) * opb(B) + beta * E
                    dF = gemm(transa, transb, alpha, dA, dB, beta, dE, 'O', algo, same_pattern=false)
                    @test F ≈ SparseMatrixCSC(dF)

                    # not same pattern
                    G = sprand(T, 25, 35, 0.4)
                    dG = SparseMatrixType(G)
                    @test_throws ErrorException("AB and C must have the same sparsity pattern.") gemm!(transa, transb, gamma, dA, dB, beta, dG, 'O', algo)
                    dG = gemm!(transa, transb, gamma, dA, dB, zero(T), dG, 'O', algo)
                    H = gamma * opa(A) * opa(B) + zero(T) * G
                    @test H ≈ SparseMatrixCSC(dG)
                end
            end
            if SparseMatrixType == CuSparseMatrixCSR
                A = sprand(T,25,10,0.2)
                B = sprand(T,10,35,0.3)
                dA = SparseMatrixType(A)
                dB = SparseMatrixType(B)
                C  = A * B
                dC = SparseMatrixType(C)
                @test_throws ArgumentError("Sparse matrix-matrix multiplication only supports transa (T) = 'N' and transb (C) = 'N'") gemm!('T', 'C', one(T), dA, dB, zero(T), dC, 'O', algo)
            end
        end
    end

    @testset "gemv $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "transa = $transa" for (transa, opa) in [('N', identity)]
            A = sprand(T,25,10,0.2)
            b = sprand(T,10,0.3)
            dA = SparseMatrixType(A)
            db = CuSparseVector(b)
            alpha = rand(T)
            y = alpha * opa(A) * b
            dy = gemv(transa, alpha, dA, db, 'O')
            @test collect(dy) ≈ y
        end
    end
end

if CUSPARSE.version() >= v"11.4.1"

    SDDMM_ALGOS = Dict(CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SDDMM_ALG_DEFAULT])

    for SparseMatrixType in keys(SDDMM_ALGOS)
        @testset "$SparseMatrixType -- sddmm! algo=$algo" for algo in SDDMM_ALGOS[SparseMatrixType]
            @testset "sddmm! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    @testset "transb = $transb" for (transb, opb) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        T <: Complex && (transa == 'C' || transb == 'C') && continue
                        mA = transa == 'N' ? 25 : 10
                        nA = transa == 'N' ? 10 : 25
                        mB = transb == 'N' ? 10 : 35
                        nB = transb == 'N' ? 35 : 10

                        A = rand(T,mA,nA)
                        B = rand(T,mB,nB)
                        C = sprand(T,25,35,0.3)

                        spyC = copy(C)
                        spyC.nzval .= one(T)

                        dA = CuArray(A)
                        dB = CuArray(B)
                        dC = SparseMatrixType(C)

                        alpha = rand(T)
                        beta = rand(T)

                        D = alpha * (opa(A) * opb(B)) .* spyC + beta * C
                        sddmm!(transa, transb, alpha, dA, dB, beta, dC, 'O', algo)
                        @test collect(dC) ≈ D
                    end
                end
            end
        end
    end
end
