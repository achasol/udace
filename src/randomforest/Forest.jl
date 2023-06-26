#=
The Forest module implements methods for the 
Random Forest Counterfactual Explanations. 

=#

module Forest

using ..ActionCandidate
using ...Feature
using ..Cost
export createForestActionCandidates, get_forest_partitions, set_forest_intervals!, generate_actions_costs_and_intervals

#Function generates the actions, costs and intervals for the Random forest counterfactual explanations. 
function generate_actions_costs_and_intervals(candidates::ForestActionCandidates, x::Vector, y::Float64; cost_type="TLPS", p=1, has_threshold=true)
    actions = construct_action_set(candidates, x, has_threshold=has_threshold)
    costs = construct_cost!(candidates, actions, x, y; cost_type, p)
    Intervals = set_forest_intervals!(candidates, x, actions)
    return actions, costs, Intervals
end



function createForestActionCandidates(X::Matrix, Y::Vector, forest; features::Features, max_candidates=50, tolerance=1e-6)
    D = size(X, 2)
    N = size(X, 1)

    trees = [t.tree_ for t in forest.estimators_]

    leaves = [findall(x -> x == -2, tree.feature) for tree in trees] #tree bug 
    L = [length(l) for l in leaves]

    candidates = ForestActionCandidates(X, Y, features, max_candidates, tolerance, N, D, [], forest.n_estimators, trees, leaves, L, [], [], [], [])

    candidates.H = construct_forest_labels(candidates)
    candidates.ancestors, candidates.regions = construct_forest_ancestors_and_regions(candidates)
    candidates.thresholds = construct_forest_thresholds(candidates)

    return candidates

end

#Function which creates the labels for the for the random forest. 
function construct_forest_labels(candidates::ForestActionCandidates)
    H = []

    for (tree, leaves, l_t) in zip(candidates.trees, candidates.leaves, candidates.L)

        falls = reshape(tree.value[leaves, :, :], (l_t, 2))
        h_t = [size(falls[i, :], 1) == 1 ? falls[i, 1] : falls[i, 2] / (falls[i, 2] + falls[i, 1]) for i in 1:l_t]
        push!(H, h_t)
    end

    return H
end


#Function constructs the thresholds for a random forest model. 
function construct_forest_thresholds(candidates::ForestActionCandidates)
    T = []
    for d in 1:candidates.D
        t_d = []
        for tree in candidates.trees
            t_d = vcat(t_d, Vector(tree.threshold[tree.feature.==d-1]))
        end
        t_d = unique(t_d)
        sort!(t_d)
        push!(T, t_d)
    end

    return T

end

#Function constructs the ancestors and regions for a Random Forest model. 
function construct_forest_ancestors_and_regions(candidates::ForestActionCandidates)
    ancestors = []
    regions = []

    a_ = nothing
    for (tree, leaves) in zip(candidates.trees, candidates.leaves)
        A = []
        R = []
        stack = [[]]
        L = [fill(-Inf, candidates.D)]
        U = [fill(Inf, candidates.D)]

        for n in range(1, tree.node_count)

            a = pop!(stack)
            l = pop!(L)
            u = pop!(U)

            if n in leaves
                push!(A, a)
                push!(R, [[l[d], u[d]] for d in 1:candidates.D])

            else
                d = tree.feature[n] + 1

                if !(d in a)
                    a_ = vcat(Array(a), [d])
                end
                push!(stack, a_)
                push!(stack, a_)
                b = tree.threshold[n]
                l_ = Array(l)
                u_ = Array(u)
                l[d] = b
                u[d] = b
                push!(U, u_)
                push!(U, u)
                push!(L, l)
                push!(L, l_)

            end


        end

        push!(ancestors, A)
        push!(regions, R)
    end

    return ancestors, regions
end

#Function which generates the indicator set I, which determines 
#Which discrete features are part of which partition. 
function set_forest_intervals!(candidates::ForestActionCandidates, x::Vector, actions::Array)
    I = []

    for t in 1:candidates.T
        I_t = []
        for l in 1:candidates.L[t]
            I_t_l = []
            for d in 1:candidates.D
                xa = x[d] .+ actions[d]
                values = []
                for i in eachindex(xa)
                    if xa[i] > candidates.regions[t][l][d][1] && xa[i] <= candidates.regions[t][l][d][2]
                        append!(values, 1.0)
                    else
                        append!(values, 0.0)
                    end
                end
                push!(I_t_l, values)
            end
            push!(I_t, I_t_l)
        end
        push!(I, I_t)
    end
    return I
end


end