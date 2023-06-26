#=
The LOF module implements all functionality related 
to the Local Outlier Factor which is used as a component in 
the Objective function. 
=#
module LOF

using LinearAlgebra: diag
using ...ActionCandidate
using ScikitLearn
@sk_import neighbors:LocalOutlierFactor
@sk_import metrics.pairwise:pairwise_kernels

export local_outlier_factor, construct_LOF_params

#Adapted from  https://github.com/BeenKim/MMD-critic/blob/master/mmd.py  
function greedy_select_prototypes(K::Matrix, candidate_indices, m; K_is_sparse=false)

    n = length(candidate_indices)

    if n != size(K, 1)
        K = K[:, candidate_indices][candidate_indices, :]
    end

    colsum = vec(2 * sum(K, dims=1))' ./ n

    selected::Array{Int} = []

    for _ in range(1, m)
        argmax = -1
        candidates = setdiff(range(1, n), selected)
        s1array = colsum[candidates]

        if length(selected) > 0
            temp = K[selected, :][:, candidates]
            s2array = 2 * sum(temp, dims=1)' + diag(K)[candidates, :]
            s2array = s2array / (length(selected) + 1)
            s1array = s1array - s2array
        else
            s1array = s1array - abs.(diag(K)[candidates, :])
        end

        argmax = candidates[findmax(s1array)[2]]
        append!(selected, argmax)

    end

    return candidate_indices[selected]
end


function select_prototypes(X::Matrix; subsample=20, kernel="rbf")
    return ifelse(subsample > 1, greedy_select_prototypes(pairwise_kernels(X, metric=kernel), range(1, size(X, 1)), subsample), range(1, size(X, 1)))
end

#Uses the LocalOutlierFactor of scikit-learn to obtain the required values for the LOF constraints and objective. 
function construct_LOF_params(candidates::ActionCandidates, y::Int64; k=10, p=2, subsample=20, kernel="rbf")
    lof = LocalOutlierFactor(n_neighbors=k, metric=ifelse(p == 1, "manhattan", "sqeuclidean"), novelty=true)
    X_lof = candidates.X[candidates.Y.==y, :]
    prototypes = select_prototypes(X_lof, subsample=subsample, kernel=kernel)
    estimated_lof = lof.fit(X_lof)

    return X_lof[prototypes, :], estimated_lof._distances_fit_X_[prototypes, k], estimated_lof._lrd[prototypes, :]
end

#Computes the lof score of a specific instance x. 
function local_outlier_factor(candidates::ActionCandidates, x::Vector, y::Float64; k=10, p=2)
    lof = LocalOutlierFactor(n_neighbors=k, metric=ifelse(p == 1, "manhattan", "sqeuclidean"), novelty=true)
    estimated_lof = lof.fit(candidates.X[candidates.Y.==y, :])
    return -1 * estimated_lof.score_samples([x])[1]
end



end