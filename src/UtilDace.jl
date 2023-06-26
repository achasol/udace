#=
The UtilDace module is capable of performing 
the experiments used to generate the results 
in my paper. 
=#

module UtilDace

include("./dataset/Feature.jl")
include("./dataset/Dataset.jl")
include("./ce_methods/LinearCE.jl")
include("./ce_methods/ForestCE.jl")
include("./preferences/Preference.jl")
include("./actions/ParetoSelection.jl")
using .Feature
using .Dataset
using .ForestCE
using .LinearCE
using .Preference
using .ParetoSelection
using DataFrames
using CSV
using ScikitLearn
import Random

#Fix the global seed 
Random.seed!(0)




@sk_import linear_model:LogisticRegression
@sk_import ensemble:RandomForestClassifier


#Function which can be used to run a comparison experiment of different methods. 
function comparison_experiment(; N=1, dataset_identifier="h", model_type="LR", K=4, time_limit=600, has_utility=false,reduce_time=false)

    #Define the Dataset Helper module 
    dataset = CreateDatasetContainer(dataset_identifier)

    if has_utility
        preferences = load_preference_weights(dataset_identifier)
    end


    #=
    The seed 9311 was obtained from trying to align the Mersene Twister random number
    generation of Python and Julia. 
    =#
    X_train, X_test, y_train, y_test = split_dataset(dataset; test_sample_size=0.25, random_state=9311)


    if model_type == "LR"
        model = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
        println("Estimating logit..\n")
        model = model.fit(X_train, y_train)
        println("Logit estimated.")

        ce = createLinearActionExtractor(model, X_train, y_train,
            features=dataset.features,
            target_name=dataset.target_name,
            target_labels=dataset.target_labels)

    elseif model_type == "RF"
        amount_of_candidates = reduce_time ? 25 : 100
        amount_of_estimators = reduce_time ? 25 : 100
        model = RandomForestClassifier(n_estimators=amount_of_estimators, max_depth=4, random_state=1)
        println("Estimating Random forest..\n")
        model = model.fit(X_train, y_train)
        println("Random forest estimated.\n")
        ce = createForestActionExtractor(model, X_train, y_train,
            features=dataset.features,
            target_name=dataset.target_name,
            target_labels=dataset.target_labels,
            max_candidates=amount_of_candidates)
    end

    #Define the proper extract method 
    extract = model_type == "LR" ? LinearCE.extract : ForestCE.extract

    predictions = model.predict(X_test)

    rejected = X_test[predictions.==1.0, :]

    N = min(size(rejected, 1), N)

    println("*Score on the testing sample: ", model.score(X_test, y_test))


    println("Generating CE's for $(N) individuals.")
    results = DataFrame([[], [], [], [], [], [], [], []], ["Applicant", "Cost", "Mahalanobis", "10-LOF", "Utility", "Best", "Time", "Alpha"])


    for (index, x) in enumerate(eachrow(rejected[1:N, :]))

        println("Looking at denied individual: $(index)")

        #Benchmark methods 
        for cost in ["TLPS", "MAD", "PCC"]
            println("Cost: $(cost)")
            actions = extract(ce, Vector(x); max_change_num=K, cost_type=cost, time_limit=time_limit, preference_weights=has_utility ? preferences[index, :] : [])
            chosen_action_index = best_action_from_pareto_set(actions)

            for (action_index, a) in enumerate(actions)
                if a != -1

                    push!(results, [index, cost, a.scores["Mahalanobis"], a.scores["10-LOF"], a.scores["Utility"], action_index == chosen_action_index, a.scores["Time"], 0.0])

                else
                    push!(results, [-1 for key in 1:8])

                end
            end

        end

        alphas =  dataset_identifier== "h" ? [1.0] : [0.01]
        #DACE/UDACE method 
        for alpha in alphas #[1.0] #German: [0.01] #Heloc: [1.0] 
            println("Cost: DACE")
            actions = extract(ce, Vector(x); max_change_num=K, alpha=alpha, cost_type="DACE", time_limit=time_limit, preference_weights=has_utility ? preferences[index, :] : [])
            chosen_action_index = best_action_from_pareto_set(actions)
            for (action_index, a) in enumerate(actions)
                if a != -1

                    push!(results, [index, "DACE", a.scores["Mahalanobis"], a.scores["10-LOF"], a.scores["Utility"], action_index == chosen_action_index, a.scores["Time"], alpha])

                else
                    push!(results, [-1 for key in 1:8])

                end
            end

        end

    end

    name = has_utility ? "util_" : ""
    CSV.write("../results/" * name * model_type * "_" * dataset.identifier * ".csv", results)



end

#Set N=50
println("Starting experiments...")
#Run the code below to reproduce the DACE results. 
#=
comparison_experiment(N=50;dataset_identifier="g",model_type="LR",has_utility=false,time_limit=600)
comparison_experiment(N=50;dataset_identifier="h",model_type="LR",has_utility=false,time_limit=600)
comparison_experiment(N=50;dataset_identifier="g",model_type="RF",has_utility=false,time_limit=600)
comparison_experiment(N=50;dataset_identifier="h",model_type="RF",has_utility=false,time_limit=600)
=#

#Run the code below to reproduce the UDACE results 
#Note that the last experiment can take several hours to complete 

#=
comparison_experiment(N=50;dataset_identifier="g",model_type="LR",has_utility=true,time_limit=600)
comparison_experiment(N=50;dataset_identifier="h",model_type="LR",has_utility=true,time_limit=600)
comparison_experiment(N=50;dataset_identifier="g",model_type="RF",has_utility=true,time_limit=600)
comparison_experiment(N=50;dataset_identifier="h",model_type="RF",has_utility=true,time_limit=600,reduce_time = true)
comparison_experiment(N=50;dataset_identifier="h",model_type="RF",has_utility=true,time_limit=600)
=#
println("Experiments finished. Check the /results folder.")
end