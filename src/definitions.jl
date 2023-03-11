"""
    AbstractAdaptCriterion

Abstract supertype for error criteria for adaptive refinement.
"""
abstract type AbstractAdaptCriterion end

"""
    SpectralError(; n=3)

Estimate the error of the interpolant by as the sum of the norm of the last `n`
coefficients of the basis functions. In higher dimensions, `n` should be increased
"""
struct SpectralError <: AbstractAdaptCriterion
    n::Int
end

SpectralError(; n=3) = SpectralError(n)

"""
    HAdaptError(; n=10)

Estimate the error of the interpolant by dividing the panel into two, computing
interpolants on the subpanels, and computing the maximum error between
interpolants at `n*p` equispaced points, where `p` is the number of points used
to compute each interpolant.
"""
struct HAdaptError <: AbstractAdaptCriterion
    n::Int
end
HAdaptError(; n=10) = HAdaptError(n)


domaintype(::Type{<:Domain{T}}) where T = T