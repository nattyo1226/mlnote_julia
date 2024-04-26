# answer
println("answer:")

using Statistics

D = [1 3; 3 6; 6 5; 8 7]
X = D[:, 1]
Y = D[:, 2]

X_mean = mean(X)
X_var = var(X, corrected=false)
Y_mean = mean(Y)
XY_cov = cov(X, Y, corrected=false)

a = XY_cov / X_var
b = Y_mean - a * X_mean

println("a = $a, b = $b")

f(x) = a * x + b

ϵ = Y .- f.(X)
println("ϵ = $ϵ")

Xϵ_cov = cov(X, ϵ, corrected=false)
println("cov(X, ϵ) = $Xϵ_cov")

fXϵ_cov = cov(f.(X), ϵ, corrected=false)
println("cov(f(X), ϵ) = $fXϵ_cov")

R2 = var(f.(X), corrected=false) / var(Y, corrected=false)
println("R^2 = $R2")

println("----------")

# check
println("check:")

using GLM, DataFrames

df = DataFrame(D, [:X, :Y])
result = lm(@formula(Y ~ X), df)

b_check, a_check = coef(result)
println("a = $a_check, b = $b_check")

pred = predict(result)
resp = response(result)

ϵ_check = resp .- pred
println("ϵ = $ϵ_check")

Xϵ_cov_check = cov(X, ϵ_check, corrected=false)
println("cov(X, ϵ) = $Xϵ_cov_check")

fXϵ_cov_check = cov(pred, ϵ_check, corrected=false)
println("cov(f(X), ϵ) = $fXϵ_cov_check")

R2_check = r2(result)
println("R^2 = $R2_check")

# plot
using Plots, LaTeXStrings

plt = plot(0:0.1:10, f, label="Regression Line", xlabel=L"x", ylabel=L"y")
scatter!(plt, X, Y, label="data")
plot!(plt, [X'; X'], [Y'; f.(X)'], ls=:dash ,label="")
scatter!(plt, X, f.(X), markershape=:x, mc=:black, label="")
