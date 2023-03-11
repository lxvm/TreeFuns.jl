"""
This package provides [`TreeFun`](@ref), which constructs a piecewise polynomial
function approximation as a tree via h-adaptation using a fixed-degree
polynomial interpolant per panel. This is in contrast to `ApproxFun`, which
utilizes p-adaptation.

    julia> using TreeFuns

    julia> f(x) = imag(inv(0.1im+sum(sin, x)))
    f (generic function with 1 method)

    julia> p = TreeFun(f, Chebyshev(0..2π)^2; criterion=SpectralError(; n=9), atol=1e-3)

"""
module TreeFuns

using LinearAlgebra: norm

using Reexport
@reexport using ApproxFun
using DomainSets: ClosedInterval, ProductDomain, dimension
import DomainSets: domaintype


export AbstractAdaptCriterion, SpectralError#, HAdaptError # to be implemented
include("definitions.jl")

export TreeFun

struct TreeFun{F} <: Function
    funtree::Vector{F}
    searchtree::Vector{Vector{Int}}
    initdiv::Int
    function TreeFun(funtree::Vector{F}, searchtree::Vector{Vector{Int}}, initdiv::Int) where {F<:Fun}
        new{F}(funtree, searchtree, initdiv)
    end
end

function (f::TreeFun)(x)
    for i in 1:f.initdiv
        fun = f.funtree[i]
        children = f.searchtree[i]
        x in domain(fun) || continue
        while true
            isempty(children) && return fun(x)
            for c in children
                x in domain(f.funtree[c]) || continue
                fun = f.funtree[c]
                children = f.searchtree[c]
                break
            end
        end
    end
    return zero(eltype(f.funtree[1].coefficients))
end


function domainsplit(d::ClosedInterval, initdiv)
    a, b = endpoints(d)
    r = range(a, b, length=initdiv+1)
    [ClosedInterval(r[i], r[i+1]) for i in 1:initdiv]
end

function domainsplit(d::ProductDomain, initdiv; dims=1:dimension(d))
    bypart = map(enumerate(components(d))) do ((i,c))
        i in dims ? domainsplit(c, initdiv) : (c,)
    end
    parts = Iterators.product(bypart...)
    vec(map(c -> ProductDomain(c), parts))
end

batcheval(f, x) = f.(x)
function batcheval!(y, f, x)
    y .= f.(x)
end

function evalerror(criterion::SpectralError, f::Fun, n, nrm)
    ncoeffs = criterion.n
    c = coefficients(f)
    sum(nrm, c[(n-ncoeffs):n])
end

function treeinterp_(criterion::SpectralError, f, s, np, atol, rtol_, nrm, maxevals, initdiv)
    rtol = rtol_ == 0 == atol ? sqrt(eps(float(domaintype(domain(s))))) : rtol_
    (rtol < 0 || atol < 0) && throw(ArgumentError("invalid negative tolerance"))
    maxevals < 0 && throw(ArgumentError("invalid negative maxevals"))
    initdiv < 1 && throw(ArgumentError("initdiv must be positive"))

    nrmtree = Float64[]
    funtree = map(domainsplit(domain(s), initdiv)) do dom
        sp = setdomain(s, dom)
        vals = batcheval(f, points(sp, np))
        push!(nrmtree, maximum(nrm, vals))
        Fun(sp, transform(sp, vals))
    end
    searchtree = fill(Int[], length(funtree))

    n = 1
    while true
        m = 0
        l = length(funtree)
        for i in n:l
            n += 1
            evalerror(criterion, funtree[i], np, nrm) > max(atol, rtol*nrmtree[i]) || continue
            fun = funtree[i]
            foreach(domainsplit(domain(space(fun)), 2)) do dom
                m += 1
                sp = setdomain(space(fun), dom)
                vals = batcheval(f, points(sp, np))
                push!(nrmtree, maximum(nrm, vals))
                push!(funtree, Fun(sp, transform(sp, vals)))
                push!(searchtree, Int[])
                push!(searchtree[i], l+m)
            end
        end
        m == 0 && break
    end
    return TreeFun(funtree, searchtree, length(funtree))
end

function treeinterp(f, s::Space; criterion=SpectralError(), n=16^dimension(domain(s)), atol=0, rtol=0, norm=norm, maxevals=typemax(Int), initdiv=1)
    treeinterp_(criterion, f, s, n, atol, rtol, norm, maxevals, initdiv)
end


"""
    TreeFun(f, s::Space; criterion=SpectralError(), n=16^dimension(domain(s)), atol=0, rtol=sqrt(eps()), norm=norm, maxevals=typemax(Int), initdiv=1)

Construct a piecewise polynomial approximation of `f` over the polynomial space
`s`. Use the error `criterion` to do the h-adaptation and use `n` points per
panel.
"""
TreeFun(args...; kwargs...) = treeinterp(args...; kwargs...)

end