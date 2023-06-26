#=
The Mahalanobis distance module contains 
methods to compute the Mahalanobis distance. 
=#
module MahalanobisDistance


using Distances: mahalanobis
using LinearAlgebra: rank, inv, eigen, eigvecs, eigvals, Symmetric, UniformScaling, Diagonal
using ....Feature: Features
using ...ActionCandidate: ActionCandidates
using ScikitLearn
@sk_import covariance:(EmpiricalCovariance, MinCovDet)

export mahalanobis_distance!, interaction_matrix

#Compute the Mahalanobis distance between two instances x1 and x2 
function mahalanobis_distance!(candidates::ActionCandidates, x_1::Vector, x_2::Vector, y::Float64)
    if candidates.cov == []
        candidates.cov, _ = interaction_matrix(ifelse(length(candidates.Y) == candidates.N, candidates.X[candidates.Y.==y, :], candidates.X), interaction_type="covariance")
    end


    return mahalanobis(x_1, x_2, Symmetric(inv(candidates.cov)))
end

#Function which determines the interaction matrix which has to be used by the mahalanobis distance 
function interaction_matrix(X; interaction_type="covariance", estimator="ML")

    if interaction_type == "covariance"
        if estimator == "ML"
            est_cov = EmpiricalCovariance(store_precision=true, assume_centered=false).fit(X)
        else
            est_cov = MinCovDet(store_precision=true, assume_centered=false, support_fraction=nothing).fit(X)
        end

        cov = Symmetric(est_cov.covariance_)

        if rank(cov) != size(X, 2)
            cov = cov + UniformScaling(1e-6)
        end

        #Perform a spectral decomposition to get the U matrix. 
        cov_inv = inv(cov)
        F = eigen(cov_inv)
        eigenvalues = eigvals(F)
        eigenvectors = eigvecs(F)
        l = Diagonal(sqrt.(eigenvalues))
        P = transpose(eigenvectors)
        U = (P' * l)'

        return cov, U
    elseif interaction_type == "correlation"
        return cor(transpose(X)) - UniformScaling(1)
    elseif interaction_type == "precomputed"
        error("interaction matrix variant precomputed not implemented")

    end

end

end