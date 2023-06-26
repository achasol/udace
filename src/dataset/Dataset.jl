#=
This module implements a Dataset helper 
which can be used to construct a features struct and 
load all data for any experiments. 
=#
module Dataset
using ..Feature
using CSV
using DataFrames
using ScikitLearn.CrossValidation


const DATASETS = ["g", "h"]

const DATASETS_NAME = Dict(
    "g" => "german",
    "h" => "fico",
)
const DATASETS_PATH = Dict(
    "g" => "data/german_credit.csv",
    "h" => "data/heloc.csv",)
const TARGET_NAME = Dict(
    "g" => "Default",
    "h" => "RiskPerformance",)
const TARGET_LABELS = Dict(
    "g" => ["No", "Yes"],
    "h" => ["Good", "Bad"],)
const FEATURE_SIZES = Dict(
    "g" => 61,
    "h" => 23,
)
const FEATURE_TYPES = Dict(
    "g" => vcat(fill("I", 7), fill("B", 54)),
    "h" => fill("I", 23),)


const FEATURE_CATEGORIES = Dict(
    "g" => [collect(range(7, 10)), collect(range(11, 15)), collect(range(16, 25)), collect(range(26, 30)), collect(range(31, 35)), collect(range(36, 39)), collect(range(40, 42)), collect(range(43, 46)), collect(range(47, 49)), collect(range(50, 52)), collect(range(53, 56)), collect(range(57, 58)), [59]],
    "h" => [],)

const FEATURE_UTILITY_CATEGORIES = Dict(
    "g" => [[32, 33, 34, 35, 36, 53, 54, 55, 56], [7, 37, 38, 39, 40, 41, 42], [8, 9, 10, 11, 27, 28, 29, 30, 31, 43, 44, 45, 46], [6, 12, 13, 14, 15, 16, 47, 48, 49], [1, 2, 3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [4, 5, 57, 58]],
    "h" => [[6, 7, 8, 9, 10, 11], [2, 3, 12, 13, 14], [15, 16, 17], [4, 18, 19, 20, 21, 22, 23], [1, 5]],)

const FEATURE_CONSTRAINTS = Dict(
    "g" => vcat(fill("", 50), fill("FIX", 3), fill("", 6), fill("FIX", 2)),
    "h" => fill("", 23),
)


#Below feature names were taken from the original source code by Kanamori et al (2020)
const FEATURE_NAMES = ["duration_in_month", "credit_amount", "installment_as_income_perc",
    "present_res_since", "age", "credits_this_bank",
    "people_under_maintenance", "account_check_status:0<=...<200DM",
    "account_check_status:<0DM", "account_check_status:>=200DM/salary_assignments_for_at_least_1year",
    "account_check_status:no_checking_account",
    "credit_history:all_credits_at_this_bank_paid_back_duly",
    "credit_history:critical_account/other_credits_existing_(not_at_this_bank)",
    "credit_history:delay_in_paying_off_in_the_past",
    "credit_history:existing_credits_paid_back_duly_till_now",
    "credit_history:no_credits_taken/all_credits_paid_back_duly",
    "purpose:(vacation-does_not_exist?)", "purpose:business",
    "purpose:car(new)", "purpose:car(used)", "purpose:domestic_appliances",
    "purpose:education", "purpose:furniture/equipment",
    "purpose:radio/television", "purpose:repairs", "purpose:retraining",
    "savings:..>=1000DM", "savings:...<100DM", "savings:100<=...<500DM",
    "savings:500<=...<1000DM", "savings:unknown/no_savings_account",
    "present_emp_since:..>=7years", "present_emp_since:...<1year",
    "present_emp_since:1<=...<4years", "present_emp_since:4<=...<7years",
    "present_emp_since:unemployed",
    "personal_status_sex:female_divorced/separated/married",
    "personal_status_sex:male_divorced/separated",
    "personal_status_sex:male_married/widowed",
    "personal_status_sex:male_single", "other_debtors:co-applicant",
    "other_debtors:guarantor", "other_debtors:none",
    "property:if_not_A121_building_society_savings_agreement/life_insurance",
    "property:if_not_A121/A122_car_or_other_not_in_attribute_6",
    "property:real_estate", "property:unknown/no_property",
    "other_installment_plans:bank", "other_installment_plans:none",
    "other_installment_plans:stores", "housing:for_free", "housing:own", "housing:rent",
    "job:management/self-employed/highly_qualified_employee/officer",
    "job:skilled_employee/official",
    "job:unemployed/unskilled-non-resident", "job:unskilled-resident",
    "telephone:none", "telephone:yes_registered_under_the_customers_name",
    "foreign_worker:no", "foreign_worker:yes"]

struct DatasetContainer
    identifier::String
    X::Matrix
    y::Vector
    target_name::String
    target_labels::Array
    features::Features

end

export CreateDatasetContainer, split_dataset, load_presplitted_dataset

#Constructor function which creates a new DatasetContainer. 
function CreateDatasetContainer(identifier)

    target_name = TARGET_NAME[identifier]
    target_labels = TARGET_LABELS[identifier]
    feature_constraints = FEATURE_CONSTRAINTS[identifier]
    feature_categories = FEATURE_CATEGORIES[identifier]
    feature_types = FEATURE_TYPES[identifier]
    feature_utility_categories = FEATURE_UTILITY_CATEGORIES[identifier]
    feature_names = identifier == "g" ? FEATURE_NAMES : []

    features = createFeatures(FEATURE_SIZES[identifier]; feature_names, feature_types, feature_categories, feature_constraints, feature_utility_categories)

    print("Loading data \n")
    df = DataFrame(CSV.File("../" * DATASETS_PATH[identifier]))
    print("Data loaded\n")
    y = Vector{Float64}(df[!, target_name])

    select!(df, Not(target_name))
    X = Matrix{Float64}(df[:, :])

    container = DatasetContainer(identifier, X, y, target_name, target_labels, features)

    return container
end


#Generate a train-test split of the dataset. 
function split_dataset(dataset::DatasetContainer; test_sample_size=0.25, random_state=nothing)
    return train_test_split(dataset.X, dataset.y; test_size=test_sample_size, random_state=random_state)

end

end