#=
The cost module contains a set of methods 
used in construction of the Objectives. 

Besides these methods also some helper functions 
for the JuMP library are contained in this file. 
=#
module Cost

include("./LOF.jl")
include("./MahalanobisDistance.jl")

import KernelDensity: kde, pdf
import Interpolations: interpolate, Gridded, Linear, linear_interpolation, Flat
using Statistics: cor, std, median

using ..ActionCandidate
using .MahalanobisDistance
using .LOF
using JuMP


export construct_utility_weights, generate_actions_and_costs, mahalanobis_distance!, local_outlier_factor, construct_LOF_params, dotspecial_expression, dotspecial_value, vecmul_expression, construct_action_set, construct_cost!



#This is a helper function for the JuMP library to deal with SparseAxisArrays. 
function dotspecial_expression(C::Array, X::JuMP.Containers.SparseAxisArray; D=nothing)
    ex = AffExpr(0.0)
    if isnothing(D)
        D = size(C, 1)
    end

    for d in 1:D
        for i in 1:length(C[d])
            add_to_expression!(ex, C[d][i], X[d, i])
        end

    end

    return ex

end

#This is a helper function for the JuMP library to deal with SparseAxisArrays, once they have a value. 
function dotspecial_value(C::Array, X::JuMP.Containers.SparseAxisArray; D=nothing)
    sum = 0
    if isnothing(D)
        D = size(C, 1)
    end


    for d in 1:D
        for i in 1:length(C[d])
            sum += C[d][i] * X[d, i]
        end

    end

    return sum

end

#This is a helper function for the JuMP library to deal with SparseAxisArrays,which are vector-shaped. 
function vecmul_expression(C::Vector, X::JuMP.Containers.SparseAxisArray)
    ex = AffExpr(0.0)
    for i in 1:length(C)
        add_to_expression!(ex, C[i], X[i])
    end
    return ex
end
#End extract 

function construct_utility_weights(D::Int, preference_weights::Array, utility_categories::Array, A::Array,)
    #Generate utility cost vector here 
    utility_weights = []
    feature_utility_category_map = Dict()
    for (index, category) in enumerate(utility_categories)
        for feature in category
            feature_utility_category_map[feature] = index
        end
    end


    for d in 1:D
        weights = [0.0 for i in eachindex(A[d])]

        if !haskey(feature_utility_category_map, d)
            push!(utility_weights, weights)
            continue
        end

        #The last option is the zero option which needs no weight. 
        for i in 1:(length(A[d])-1)
            weights[i] = -1.0 * preference_weights[feature_utility_category_map[d]] #Add scaling factor depending on action distance 
        end
        push!(utility_weights, weights)
    end
    return utility_weights
end



#Fetches the actions and costs for the Logistic regression model. 
function generate_actions_and_costs(candidates::LinearActionCandidates, x::Vector, y::Float64; cost_type="TLPS", p=1)
    actions = construct_action_set(candidates, x)
    costs = construct_cost!(candidates, actions, x, y; cost_type, p)
    return actions, costs
end



#Generates an interpolated CDF based on the kernel-density of the data X_d. 
function cumulative_distribution_function(x_d, X_d; l_buffer=1e-6, r_buffer=1e-6)
    U = kde(X_d)
    density = pdf(U, x_d)
    raw_cdf = cumsum(density)
    total_cdf = last(raw_cdf) + l_buffer + r_buffer
    cdf = (l_buffer .+ raw_cdf) / total_cdf

    return interpolate((x_d,), cdf, Gridded(Linear()))
end



#Function which uses the chosen cost type to determine the weights which must be used in the objective. 
function retrieve_feature_weight(candidates::ActionCandidates; cost_type="uniform")
    weights = ones(Float64, candidates.D)

    if cost_type == "MAD"
        for d in range(1, candidates.D)
            A = candidates.X[:, d]
            weight = median(abs.(A .- median(A))) / 0.67449 #Correct for normal scale 
            if candidates.features.types[d] == "B" || abs(weight) < candidates.tolerance
                weights[d] = std(A .* 1.4826)
            else
                weights[d] = (1 / weight)
            end
        end
    elseif cost_type == "standard"
        weights = std(candidates.X)^-1
    elseif cost_type == "PCC" && length(candidates.Y) == candidates.N
        for d in range(1, candidates.D)
            weights[d] = abs(cor(candidates.X[:, d], candidates.Y))
        end
    elseif cost_type == "normalize"
        weights = (findmax(candidates.X, 1) - findmin(candidates.X, 1))^-1


    end

    return weights

end



#Constructs the cost used in the objective function based on the cost type. 
function construct_cost!(candidates::ActionCandidates, actions::Array, x::Vector, y::Float64; cost_type="TLPS", p=1)

    costs = []
    X_ub = findmax(candidates.X, dims=1)[1]
    X_lb = findmin(candidates.X, dims=1)[1]
    steps = [ifelse(candidates.features.types[d] == 'C', (X_ub[d] - X_lb[d]) / candidates.max_candidates, 1) for d in range(1, candidates.D)]

    if cost_type == "TLPS"
        Q = nothing
        grids = [range(X_lb[d], X_ub[d] + steps[d] - 1, step=steps[d]) for d in range(1, candidates.D)]
        if isnothing(Q)
            Q = [candidates.features.constraints[d] == "FIX" ? nothing : cumulative_distribution_function(grids[d], candidates.X[:, d]) for d in range(1, candidates.D)]
        end
        for d in range(1, candidates.D)

            if isnothing(Q[d])
                push!(costs, [])
            else
                Q_d = Q[d]
                Q_0 = Q_d(x[d])

                Q_xa(a) = ((x[d] + a) > grids[d][end]) ? 1 - 1e-6 : ((x[d] + a) < grids[d][1] ? 1e-6 : Q_d(x[d] + a))
                push!(costs, [abs.(log2((1 - Q_xa(a)) / (1 - Q_0))) for a in actions[d]])

            end

        end

    elseif cost_type == "DACE"

        candidates.cov, B = interaction_matrix(ifelse(length(candidates.Y) == candidates.N, candidates.X[candidates.Y.==y, :], candidates.X), interaction_type="covariance")

        C = retrieve_feature_weight(candidates, cost_type="uniform")

        for d in range(1, candidates.D)
            cost_d = []
            for j in range(1, candidates.D)
                push!(cost_d, [C[d] * B[d, j] * a for a in actions[j]])
            end
            push!(costs, cost_d)
        end
    else
        weights = retrieve_feature_weight(candidates, cost_type=cost_type)

        if cost_type == "PCC"
            p = 2
        end

        for d in range(1, candidates.D)
            push!(costs, (weights[d] * abs.(actions[d]) .^ p))
        end
    end


    return costs
end


end