const _allowscalar = Ref(true)

allowscalar(flag = true) = (_allowscalar[] = flag)

function assertscalar(op = "Operation")
  _allowscalar[] || error("$op is disabled")
  return
end

Base.IndexStyle(::Type{<:CuArray}) = IndexLinear()

function _getindex(xs::CuArray{T}, i::Integer) where T
  buf = Mem.view(unsafe_buffer(xs), (i-1)*sizeof(T))
  return Mem.download(T, buf)[1]
end

function Base.getindex(xs::CuArray{T}, i::Integer) where T
  ndims(xs) > 0 && assertscalar("scalar getindex")
  _getindex(xs, i)
end

function Base.getindex(xs::CuArray{T}, i::UnitRange) where T
  CuVector{T}(xs.buf, xs.offset+(i.start-1)*sizeof(T), (i.stop-i.start+1,))
end

function _setindex!(xs::CuArray{T}, v::T, i::Integer) where T
  buf = Mem.view(unsafe_buffer(xs), (i-1)*sizeof(T))
  Mem.upload!(buf, T[v])
end

function Base.setindex!(xs::CuArray{T}, v::T, i::Integer) where T
  assertscalar("scalar setindex!")
  _setindex!(xs, v, i)
end

Base.setindex!(xs::CuArray, v, i::Integer) = xs[i] = convert(eltype(xs), v)
