#= 
This module implements an Action struct which can be used 
to calculate statistics and make it easy to print an action
to the terminal. 
=#
module Actions
using ...Feature

mutable struct Action
    x::Vector
    a::Vector
    scores::Dict
    target_name::String
    target_labels::Vector
    label_before::Int
    label_after::Int
    features::Features

end

export Action, createAction


#Constructs an action 
function createAction(x::Vector, a::Vector; scores::Dict=Dict(), target_name::String="Output", target_labels::Vector=["Good", "Bad"], label_before::Int=2, label_after::Int=1, features::Features)
    a = Action(x, a, scores, target_name, target_labels, label_before, label_after, features)
    return a
end

#Function adapted from the source code of Kanamori et al, can be used to print an action nicely. 
function print_action(a::Action)
    s = "#"
    s = s * "Action ($(a.target_name): $(a.target_labels[a.label_before]) -> $(a.target_labels[a.label_after]):\n"

    inverse_feature_categories = get_inverse_feature_categories(a.features, length(a.x))

    for d in eachindex(a.a)
        if abs(a.a[d]) < 1e-8
            continue
        end
        number = "* "

        g = inverse_feature_categories[d]
        if g == -1
            s = s * "$(number) $(a.features.names[d]): $(a.x[d]) -> $(a.x[d]+a.a[d]) ($(a.a[d]))\n "

        elseif a.x[d] != 1
            category_name, next = split(a.features.names[d], ":")
            category = a.features.categories[g] .+ 1
            previous = split(a.features.names[category[findfirst(a.x[category] .== 1)]], ":")[2]
            s = s * "$(number) $(category_name): $(previous) -> $(next) \n"
        end
    end

    if length(a.scores) > 0
        s = s * "* Scores: \n"
        for item in a.scores
            s = s * "\t* $(item.first): $(item.second)\n"
        end
    end

    print(s)
end



#overwrite the default print method of an action 
Base.show(io::IO, a::Action) = print_action(a)

end