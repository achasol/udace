#=
The LinearCE module can generate 
counterfactual explanations for the Logistic regression model. 
=#
module LinearCE

include("../actions/ActionCandidate.jl")
include("../costs/Cost.jl")
include("../actions/Action.jl")
using ..Feature

using .Actions: Action, createAction
using JuMP
using HiGHS
import MultiObjectiveAlgorithms as MOA

#Cost type not used! 
using .Cost
using .ActionCandidate


export extract, createLinearActionExtractor

mutable struct LinearActionExtractor
    model::Any
    coef::Array
    intercept::Float64
    X::Matrix{Float64}
    y::Vector{Float64}
    N::Int
    D::Int
    features::Features
    max_candidates::Int
    action_candidates::ActionCandidates
    tolerance::Float64
    target_name::String
    target_labels::Vector{String}


end





function extract(extractor::LinearActionExtractor, x::Vector; W=[], max_change_num=4, cost_type::String, alpha=0.0, preference_weights::Vector=[], n_neighbours=10,
    p_neighbours=2, subsample=20, solver="HiGHS", time_limit=180, log_stream=false)

    y = extractor.model.predict([x])[1]
    K = min(max_change_num, extractor.D)

    A, C = generate_actions_and_costs(extractor.action_candidates, x, y; cost_type=cost_type)

    #Can be joined into a single for loop. 
    lb = [findmin(A_d)[1] for A_d in A]
    ub = [findmax(A_d)[1] for A_d in A]

    sparse_actions = get_sparse_actions(extractor.D, A, get_flat_feature_categories(extractor.features))

    has_lof_cost_component = alpha > 0
    has_utility_objective = length(preference_weights) > 0  

    if has_lof_cost_component
        X_lof, k_dists, lrds = construct_LOF_params(extractor.action_candidates, round(Int, 1 - y); k=n_neighbours, p=p_neighbours, subsample=subsample)
        N_lof = size(X_lof, 1)
        lrds = vec(lrds) 
        k_dists = vec(k_dists) 

        C_lof = [[[(abs(x_lof[d] - (x[d] + a)))^p_neighbours for a in A[d]] for d in range(1, extractor.D)] for x_lof in eachrow(X_lof)]
        U_lof = [sum([findmax(c[d])[1] for d in range(1, extractor.D)]) for c in C_lof]
    end



    if has_utility_objective
        utility_weights = construct_utility_weights(extractor.D, preference_weights, extractor.features.utility_categories, A)
    end

    optimizer = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => log_stream)
    if has_utility_objective
        model = JuMP.Model(() -> MOA.Optimizer(optimizer))
        set_attribute(model, MOA.Algorithm(), MOA.KirlikSayin())
        set_attribute(model, MOI.TimeLimitSec(), time_limit)
    else
        model = Model(optimizer;)
        set_time_limit_sec(model, time_limit)
    end




    @variable(model, lb[d] <= a[d=1:extractor.D] <= ub[d])

    @variable(model, pie[d=1:extractor.D, i=1:length(A[d])], binary = true)

    @variable(model, x[d] + lb[d] <= xi[d=1:extractor.D] <= x[d] + ub[d])
    if cost_type == "DACE"
        @variable(model, dist[d=1:extractor.D] >= 0)
    end


    if has_lof_cost_component

        @variable(model, 0 <= rho[n=1:N_lof] <= U_lof[n])
        @variable(model, mu[n=1:N_lof], binary = true)
    end



    #Objective function
    if cost_type == "DACE"



        if has_lof_cost_component

            @expression(model, effort_expr, sum(dist) + alpha * sum([lrds[i] * rho[i] for i in eachindex(rho)]))
            @constraint(model, c1, effort_expr >= 0)



        else
            @expression(model, effort_expr, sum(dist))
            @constraint(model, c1, effort_expr >= 0)


        end


        if has_utility_objective
            @constraint(model, c_upper_effort, effort_expr <= 30)
        end

        @constraint(model, cp[d=1:extractor.D], dist[d] - dotspecial_expression(C[d], pie) >= 0)
        @constraint(model, cz[d=1:extractor.D], dist[d] + dotspecial_expression(C[d], pie) >= 0)



    else
        if has_lof_cost_component
            @expression(model, effort_expr, dotspecial_expression(C, pie) + alpha * sum([lrds[i] * rho[i] for i in eachindex(rho)]))
            @constraint(model, C_basic_cost, effort_expr >= 0)
        else
            @expression(model, effort_expr, dotspecial_expression(C, pie))
            @constraint(model, C_basic_cost, effort_expr >= 0)
        end
    end

    if has_utility_objective
        @expression(model, utility_expr, dotspecial_expression(utility_weights, pie))
        @objective(model, Min, [effort_expr, utility_expr])
    else
        @objective(model, Min, effort_expr)
    end

    #Note pie has no order but inside sum is fine 
    @constraint(model, C_basic_pi[d=1:extractor.D], sum(pie[d, :]) == 1)

    if K >= 1
        @constraint(model, C_basic_sparsity, dotspecial_expression(sparse_actions, pie) <= K)
    end

    #Add one for proper indexing. 
    @constraint(model, C_basic_category[G in extractor.features.categories], sum([a[d+1] for d in G]) == 0)



    if y == 0
        @constraint(model, C_basic_alter, (extractor.coef * xi) .+ (extractor.intercept - 1e-8) >= 0)
    else
        @constraint(model, C_basic_alter, (extractor.coef * xi) .+ (extractor.intercept + 1e-8) <= 0)
    end

    @constraint(model, C_basic_act_[d=1:extractor.D], a[d] - vecmul_expression(A[d], pie[d, :]) == 0)


    @constraint(model, C_basic_linmodel[d=1:extractor.D], xi[d] - a[d] == x[d])

    if has_lof_cost_component
        #ADD LOF constraints 
        @constraint(model, C_lof_1nn, sum(mu) == 1)

        @constraint(model, C_lof_kdist_[n in 1:N_lof], rho[n] - k_dists[n] * mu[n] >= 0)
        @constraint(model, C_lof_dist_[n in 1:N_lof], rho[n] - (U_lof[n] * mu[n]) - dotspecial_expression(C_lof[n], pie) >= -1 * U_lof[n])


        for n in 1:N_lof
            for m in 1:N_lof
                if m == n
                    continue
                end
                tmp = [[c_n - c_m for (c_n, c_m) in zip(C_lof[n][d], C_lof[m][d])] for d in 1:extractor.D]
                @constraint(model, U_lof[n] * mu[n] + dotspecial_expression(tmp, pie) <= U_lof[n])
            end
        end



    end


    s = time()
    optimize!(model)
    t = time() - s


    println("results:", result_count(model))
    #println(solution_summary(model))

    if termination_status(model) == OPTIMAL
        println("An Optimal solution has been found.")

    elseif termination_status(model) == TIME_LIMIT && has_values(model)
        println("Solution is suboptimal due to a time limit, but a primal solution is available")

    else
        return [-1]
    end






    actions = []

    number_of_results = has_utility_objective ? result_count(model) : 1
    for result in 1:number_of_results
        obj = objective_value(model; result=result)

        if cost_type == "DACE"
            act_cost = sum([value(d; result=result) for d in dist])
        else

            act_cost = dotspecial_value(C, value.(pie; result=result))
        end

        if has_lof_cost_component
            lof_cost = sum([l * value(r; result=result) for (l, r) in zip(lrds, rho)])
        end

        action_values = [sum([A[d][i] * round(value(pie[d, i]; result=result)) for i in 1:length(A[d])]) for d in 1:extractor.D]


        scores = Dict()
        scores["Time"] = t
        scores["alpha"] = alpha
        scores["Objective"] = obj
        scores["Cost $(ifelse(cost_type == "DACE","l1-Mahal",cost_type))"] = act_cost
        if has_lof_cost_component
            scores["Cost (LOF)"] = lof_cost
        end
        scores["Utility"] = has_utility_objective ? -1.0 * obj[2] : -1.0
        scores["Mahalanobis"] = mahalanobis_distance!(extractor.action_candidates, x, x + action_values, round(y))
        scores["10-LOF"] = local_outlier_factor(extractor.action_candidates, x + action_values, round(1 - y); k=10)

        chosen_action = createAction(x, action_values; scores=scores, target_name=extractor.target_name, target_labels=extractor.target_labels,
            label_before=round(Int, y) + 1, label_after=round(Int, 1 - y) + 1, extractor.features)

        push!(actions, chosen_action)
    end

    return actions
end




function createLinearActionExtractor(model, X::Matrix{Float64}, y::Vector{Float64}; features::Features, max_candidates=100, tolerance=1e-6, target_name="Output", target_labels=["Good", "Bad"])
    N, D = size(X)

    action_candidates = createLinearActionCandidates(X, y; features, max_candidates, tolerance)
    linear_action_extractor = LinearActionExtractor(model, model.coef_, model.intercept_[1], X, y, N, D, features, max_candidates, action_candidates, tolerance, target_name, target_labels)

    return linear_action_extractor
end


end