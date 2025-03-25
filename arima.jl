using CSV, DataFrames, CustomARIMA
using BenchmarkTools

# Load data
airpass = CSV.read("airline_passengers.csv", DataFrame)
y = airpass.Passengers

# Fit ARIMA model
@btime CustomARIMA.arima($y, 2, 1, 1)
model = CustomARIMA.arima(y, 2, 1, 1)

# Print model details
println("AR coefficients: ", model.ar_coef)
println("MA coefficients: ", model.ma_coef)
println("Residual variance: ", model.sigma2)

# Forecast
@btime CustomARIMA.forecast($model, 12)
forecasts = CustomARIMA.forecast(model, 12)
println("Forecasts: ", forecasts)

# Prepare DataFrame for plotting
Y_df = DataFrame(ds=collect(1:length(y)), Passengers=y)
Y_hat_df = DataFrame(ds=collect(length(y)+1:length(y)+12), Passengers=forecasts)
CustomARIMA.plot_series(Y_df, Y_hat_df)
