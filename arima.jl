using CSV, DataFrames,CustomARIMA
using BenchmarkTools
airpass = CSV.read("airline_passengers.csv", DataFrame)
y = airpass.Passengers
@btime CustomARIMA.arima($y, 2, 1, 1)
model = CustomARIMA.arima(y, 2, 1, 1)
println("AR coefficients: ", model.ar_coef)
println("MA coefficients: ", model.ma_coef)
println("Residual variance: ", model.sigma2)
@btime CustomARIMA.forecast($model, 12)
forecasts = CustomARIMA.forecast(model, 12)
println("Forecasts: ", forecasts)