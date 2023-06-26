#=
This module implements the method to select pareto efficient solutions which
maximize the price performance ratio described by the paper 

Multi-objective optimization: A method for selecting the optimal
solution from Pareto non-inferior solutions, 2017 

By Nuo Wang, Wei-jie Zhao, Nuan Wu, Di Wu
=#

module ParetoSelection 


export best_action_from_pareto_set


#=
Returns the index of the best action from the set of pareto-efficient actions. 
=#
function best_action_from_pareto_set(actions::Array)

    
    if size(actions,1) == 0
        return -1 
    elseif size(actions,1) == 1 
        return 1 
    end


    objectives = []
    for a in actions 
        push!(objectives,a.scores["Objective"])
    end 

    sort!(objectives, by = first, rev = false)
    average_variability = []
    diff_prev = [objectives[2][2] - objectives[1][2],objectives[2][1] - objectives[1][1]]
    push!(average_variability,[diff_prev[1]/diff_prev[2],diff_prev[2]/diff_prev[1]])

    for i in 2:(length(objectives)-1)

        diff_prev = [objectives[i][2] - objectives[i-1][2],objectives[i][1] - objectives[i-1][1]]
        diff_next = [objectives[i+1][2] - objectives[i][2],objectives[i+1][1] - objectives[i][1]]

        push!(average_variability,[0.5*(diff_prev[1]/diff_prev[2] + diff_next[1]/diff_next[2]),0.5*(diff_prev[2]/diff_prev[1] + diff_next[2]/diff_next[1])])

    end

    diff_next = [objectives[end][2] - objectives[end-1][2],objectives[end][1] - objectives[end-1][1]]
    push!(average_variability,[diff_next[1]/diff_next[2],diff_next[2]/diff_next[1]])

    sensitivity_ratios = []
    for i in eachindex(objectives)
        push!(sensitivity_ratios,[average_variability[i][1]/objectives[i][1],average_variability[i][2]/objectives[i][2]])
    end 

    sums = [0.0,0.0]
    for ratio in sensitivity_ratios

        sums[1] += ratio[1]
        sums[2] += ratio[2]
    end 

    epsilons = []
    epsilon_abs_diff = []
    for ratio in sensitivity_ratios
        push!(epsilons,[ratio[1]/sums[1],ratio[2]/sums[2]])
        append!(epsilon_abs_diff,abs((ratio[1]/sums[1]) - (ratio[2]/sums[2])))
    end 

    good_solution_index = findmin(epsilon_abs_diff)[2]

    println("The good solution is: $(objectives[good_solution_index])")
    return good_solution_index 

end 



end 