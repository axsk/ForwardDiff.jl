using ForwardDiff: Dual, value

function dualarraytomatrix{N,T}(x::Vector{Dual{N,T}})
  n = length(x)
  m = Matrix{T}(n,N+1)
  for i=1:n
    m[i,1] = value(x[i])
    for j=1:N
      m[i,j+1] = x[i].partials.values[j]
    end
  end
  m
end

function matrixtodualarray{T}(x::Matrix{T})
  m,n = size(x)
  reinterpret(Dual{n-1,T}, x', (m,))
end

function Base.:*{T,N}(A::Matrix{T}, d::Vector{Dual{N,T}})
  v = dualarraytomatrix(d)
  matrixtodualarray(A*v)
end


function testjacobian(n=256^2,m=300)
  A = rand(n,m)
  x = rand(m)
  g = ForwardDiff.jacobian(x->A*x, x)
end


function testbigmult(n=256^2, m=300; c=10)
  A = rand(n,m)
  x = rand(m)
  ForwardDiff.gradient(x->sum(A*x), x, ForwardDiff.Chunk{c}())
end
