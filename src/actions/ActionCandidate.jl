#=
This module creates an actionCandidate set 
for both the Linear and Random forest model. 
=#

module ActionCandidate

using ...Feature


const DECREASE = "DEC"
const INCREASE = "INC"
const FIXED = "FIX"


abstract type ActionCandidates end
mutable struct LinearActionCandidates <: ActionCandidates
    X::Matrix
    Y::Vector
    features::Features
    max_candidates::Int
    tolerance::Float64
    N::Int
    D::Int
    cov::Array

end

mutable struct ForestActionCandidates <: ActionCandidates
    X::Matrix
    Y::Vector
    features::Features
    max_candidates::Int
    tolerance::Float64
    N::Int
    D::Int
    cov::Array
    T::Int
    trees::Array
    leaves::Array
    L::Vector
    H::Vector
    ancestors::Array
    regions::Array
    thresholds::Array

end

export ActionCandidates, LinearActionCandidates, ForestActionCandidates, createLinearActionCandidates, construct_action_set, get_sparse_actions


function createLinearActionCandidates(X::Matrix, Y::Vector; features::Features, max_candidates=50, tolerance=1e-6)
    D = size(X, 2)
    N = size(X, 1)
    candidates = LinearActionCandidates(X, Y, features, max_candidates, tolerance, N, D, [])
    return candidates

end

#Function responsible for creation of the discrete set of actions for the Logistic regression model. 
function construct_action_set(candidates::LinearActionCandidates, x::Vector)
    actions = Any[]

    X_lowerbound = findmin(candidates.X, dims=1)[1]
    X_upperbound = findmax(candidates.X, dims=1)[1]
    steps = [ifelse(candidates.features.types[d] == 'C', (X_upperbound[d] - X_lowerbound[d]) / candidates.max_candidates, 1) for d in range(1, candidates.D)]


    for d in range(1, candidates.D)
        if candidates.features.types[d] == "B"
            if (candidates.features.constraints[d] == DECREASE && x[d] == 0) || (candidates.features.constraints[d] == INCREASE && x[d] == 1)
                push!(actions, [0])
            else
                push!(actions, [1 - 2 * x[d], 0])
            end

        elseif candidates.features.constraints[d] == FIXED || steps[d] < candidates.tolerance
            push!(actions, [0])
        else
            if candidates.features.constraints[d] == INCREASE
                stop = X_upperbound[d] + steps[d]
                start = x[d] + steps[d]
            elseif candidates.features.constraints[d] == DECREASE
                stop = x[d]
                start = X_lowerbound[d]
            else
                stop = X_upperbound[d] + steps[d]
                start = X_lowerbound[d]

            end
            A_d = range(start, stop - 1, step=steps[d]) .- x[d]
            A_d = A_d[abs.(A_d).>candidates.tolerance]

            if length(A_d) > candidates.max_candidates
                A_d = [A_d[trunc(Int, a)] for a in range(1, length(A_d), length=candidates.max_candidates)]

            end

            append!(A_d, 0)
            push!(actions, A_d)

        end
    end

    return actions
end

#Function which constructs the discrete set of actions for the Random Forest model. 
function construct_action_set(candidates::ForestActionCandidates, x::Vector; has_threshold=true)

    actions = Any[]

    X_lowerbound = findmin(candidates.X, dims=1)[1]
    X_upperbound = findmax(candidates.X, dims=1)[1]
    steps = [ifelse(candidates.features.types[d] == 'C', (X_upperbound[d] - X_lowerbound[d]) / candidates.max_candidates, 1) for d in range(1, candidates.D)]

    for d in range(1, candidates.D)
        if candidates.features.types[d] == "B"
            if (candidates.features.constraints[d] == INCREASE && x[d] == 1) || (candidates.features.constraints[d] == DECREASE && x[d] == 0)
                push!(actions, [0])
            else
                push!(actions, [1 - 2 * x[d], 0])
            end
        elseif candidates.features.constraints[d] == FIXED || steps[d] < candidates.tolerance
            push!(actions, [0])
        else
            if has_threshold

                A_d = candidates.features.types[d] == "I" ? round.(candidates.thresholds[d] .- x[d]) : candidates.thresholds[d] .- x[d]
                A_d[A_d.>=0] = A_d[A_d.>=0] .+ (candidates.features.types[d] == "C" ? candidates.tolerance : 1.0)
                if !(0 in A_d)
                    append!(A_d, 0)
                end
                if candidates.features.constraints[d] == INCREASE
                    A_d = A_d[A_d.>=0]
                elseif candidates.features.constraints[d] == DECREASE
                    A_d = A_d[A_d.<=0]

                end
            else
                if candidates.features.constraints[d] == INCREASE
                    stop = X_upperbound[d] + steps[d]
                    start = x[d] + steps[d]

                elseif candidates.features.constraints[d] == DECREASE
                    stop = x[d]
                    start = X_lowerbound[d]

                else
                    stop = X_upperbound[d] + steps[d]
                    start = X_lowerbound[d]
                end
                A_d = range(start, stop - 1, step=steps[d]) .- x[d]

                A_d = A_d[abs.(A_d).>candidates.tolerance]

                if length(A_d) > candidates.max_candidates
                    A_d = [A_d[trunc(Int, a)] for a in range(1, length(A_d), length=candidates.max_candidates)]

                end

                append!(A_d, 0)
            end



            push!(actions, A_d)



        end
    end

    return actions
end

#Function returns the actions which are part of a certain feature category 
#Used to formulate the sparsity constraint. 
function get_sparse_actions(D::Int, A::Array, flat_feature_categories::Array)
    non_zeros = []

    for d in 1:D
        non_zeros_d = [1 for _ in range(1, length(A[d]))]
        for i in 1:length(A[d])
            if d in flat_feature_categories
                if A[d][i] <= 0
                    non_zeros_d[i] = 0
                end
            elseif A[d][i] == 0
                non_zeros_d[i] = 0
            end
        end
        push!(non_zeros, non_zeros_d)
    end
    return non_zeros
end


end