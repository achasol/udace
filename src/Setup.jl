#=
This file can be used to easily install all dependencies of this project. 
Naturally 
=#

import Pkg;
Pkg.instantiate()
#Packages used by UDACE 
Pkg.add("KernelDensity")
Pkg.add("Interpolations")
Pkg.add("Distances")
Pkg.add("ScikitLearn")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("JuMP")
Pkg.add("HiGHS")
Pkg.add("MultiObjectiveAlgorithms")
Pkg.add("PyCall")

#Packages used by Preference simulation 
Pkg.add("Distributions")
Pkg.add("DataStructures")
Pkg.add("Turing")
Pkg.add("Optim")
Pkg.add("StatsBase")