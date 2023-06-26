#=
This module deals with all synthethic preference generation.
1. It simulates the data  
2. Estimates a Bradley-Terry model on the data 
3. Stores the estimated data into CSV files for further processing. 
=#

module Preference


import Random
using Distributions
using DataStructures
using LinearAlgebra
using Turing
using Optim
using DataFrames
using StatsBase
using CSV

Random.seed!(1)

export load_preference_weights

#Function which loads preferences weights from CSV files for the experiments. 
function load_preference_weights(dataset_identifier::String)

    if dataset_identifier == "h"
        df = DataFrame(CSV.File("../data/preferences_D_5_N_1000.csv"))
    else
        df = DataFrame(CSV.File("../data/preferences_D_6_N_1000.csv"))
    end

    df = df[df.Type.=="estimated", dataset_identifier == "h" ? ["C1", "C2", "C3", "C4", "C5"] : ["C1", "C2", "C3", "C4", "C5", "C6"]]

    return Matrix{Float64}(df)
end



#Function which prints preferences to the terminal. 
function print_preference(latent_preferences::Array)

    signs = [">" for i in 1:length(latent_preferences)] #
    result = collect(Iterators.flatten(zip(latent_preferences, signs)))
    append!(result, latent_preferences[end])
    println(result)
end


#Preference simulation function. 
#Uses a priority queue to provide uniform draws over the pairs. 
function generate_preferences(N::Int,)
    latent_preferences = Random.randperm(N)
    print_preference(latent_preferences)

    pq = PriorityQueue{Int,Int}(Base.Order.Forward)
    count = Dict()
    position = Dict()
    for (index, action) in enumerate(latent_preferences)
        enqueue!(pq, action, 0)
        count[action] = 0
        position[action] = index
    end

    comparisons = []
    effort_distribution = Beta(2, 32)
    fatigue_distribution = Beta(10, 2)

    theta = rand(effort_distribution, 1)[1] #Effort parameter 
    initial_e = 1 #Take random draw
    final_e = rand(fatigue_distribution, 1)[1] #Take random draw larger than initial 
    judge = Bernoulli(initial_e)
    geom = Geometric(theta) #Capped at the number of pairs. 
    trials = min(N + rand(geom, 1)[1], N * (N - 1) * 0.5)#Model the amount of comparisons made using a Geometric distribution. 


    for i in 1:trials
        action1 = dequeue!(pq)
        action2 = dequeue!(pq)
        count[action1] += 1
        count[action2] += 1

        flip = rand(judge, 1)[1]

        result = []


        if position[action1] < position[action2]
            result = [action1, action2]
        else
            result = [action2, action1]
        end

        if flip == 0
            reverse!(result)
        end

        push!(comparisons, result)
        enqueue!(pq, action1, count[action1])
        enqueue!(pq, action2, count[action2])

        e = (final_e - initial_e) * sqrt(i / trials) + initial_e
        judge = Bernoulli(e)
    end

    return comparisons, latent_preferences
end



#This function specifies the Bradley-Terry model used to 
#Estimate on the generated preferences. 
@model function bradley_terry(y::Array, options::Int, data::Vector{Any})

    #action ~ truncated(Normal(0.0,1.0);lower=0) 
    action_preferences ~ Product([LogNormal(1, 1) for _ in 1:options]) #truncated(Normal(1,1);lower= 0)

    for (n, entry) in enumerate(data)
        y[n] ~ BernoulliLogit(action_preferences[entry[1]] - action_preferences[entry[2]])
    end



end

#This method samples the preferences for N users for|C| categories. 
function sample_preferences(C::Int, N::Int)

    preferences = DataFrame([[] for i in 1:(C+2)], vcat(["Applicant", "Type"], ["C$(i)" for i in 1:C]))
    for i in 1:N
        observed_comparisons, latent_preferences = generate_preferences(C)
        model = bradley_terry(fill(1, length(observed_comparisons)), C, observed_comparisons)
        map_estimate = optimize(model, MAP())
        categories = collect(1:C)
        estimated_preferences = map_estimate.values
        observed = [a[2] for a in sort(collect(zip(estimated_preferences, categories)); by=first, rev=true)]

        latent_rank = [0 for c in 1:C]
        observed_rank = [0 for c in 1:C]
        for (index, value) in enumerate(latent_preferences)
            latent_rank[value] = index
        end
        for (index, value) in enumerate(observed)
            observed_rank[value] = index
        end

        push!(preferences, vcat([i, "latent_rank"], latent_rank))
        push!(preferences, vcat([i, "observed_rank"], observed_rank))
        push!(preferences, vcat([i, "estimated"], estimated_preferences))

    end

    CSV.write("../../data/" * "preferences_" * "D_$(C)_N_$(N)" * ".csv", preferences)
    return preferences
end


#Generate preferences for HElOC and GERMAN datasets 
#sample_preferences(6,1000)
#sample_preferences(5,1000)


end