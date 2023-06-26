#=
The Feature module contains a variety of 
helper methods which make dealing with features of a dataset 
easier. 
=#

module Feature
struct Features
    names::Vector
    types::Vector
    categories::Vector
    constraints::Vector
    utility_categories::Vector
end

export Features, createFeatures, get_flat_feature_categories, get_inverse_feature_categories

#Function which fills the features with default values. 
function fill_default_features!(D::Int, feature_names, feature_types, feature_constraints)
    feature_names = ifelse(length(feature_names) == D, feature_names, ["x_$(d)" for d in range(1, D)])
    feature_types = ifelse(length(feature_types) == D, feature_types, ["C" for d in range(1, D)])
    feature_constraints = ifelse(length(feature_constraints) == D, feature_constraints, ["" for d in range(1, D)])
end

#Constructor method which creates the features. 
function createFeatures(amount; feature_names::Vector=[], feature_types::Vector=[], feature_categories::Vector=[], feature_constraints::Vector=[], feature_utility_categories::Vector=[])
    fill_default_features!(amount, feature_names, feature_types, feature_constraints)
    f = Features(feature_names, feature_types, feature_categories, feature_constraints, feature_utility_categories)
    return f
end

#Function which returns a flattended array of the feature categories. 
function get_flat_feature_categories(features::Features)
    return vcat(features.categories...)
end

#Function which returns the inverse feature categories. (hashmap)
function get_inverse_feature_categories(features::Features, D::Int)
    inverse_feature_categories = []
    for d in range(1, D)
        g = -1
        if features.types[d] == "B"
            for (index, category) in enumerate(features.categories)
                if d in category
                    g = index
                    break
                end
            end
        end
        append!(inverse_feature_categories, g)
    end
    return inverse_feature_categories
end

end