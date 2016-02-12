isdefined(Base, :__precompile__) && __precompile__()

module ForwardDiff

import Calculus
import NaNMath

const IS_MULTITHREADED_JULIA = VERSION >= v"0.5.0-dev+923" && Base.Threads.nthreads() > 1
const NTHREADS = IS_MULTITHREADED_JULIA ? Base.Threads.nthreads() : 1
const AUTO_DEFINED_UNARY_FUNCS = map(first, Calculus.symbolic_derivatives_1arg())
const NANMATH_FUNCS = (:sin, :cos, :tan, :asin, :acos, :acosh,
                       :atanh, :log, :log2, :log10, :lgamma, :log1p)

include("Partials.jl")
include("DiffNumber.jl")
include("cache.jl")
include("api.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")

end # module
