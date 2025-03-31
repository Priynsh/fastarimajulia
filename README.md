# Efficient Julia Implementation of ARIMA

This repository provides an efficient Julia implementation of the ARIMA (AutoRegressive Integrated Moving Average) model, designed to mirror the functionality of the popular statsforecast library in Python. The implementation leverages multithreading and the L-BFGS optimizer for fast and accurate parameter estimation.

## Key Features
- **Multithreading:** The implementation uses Julia's native multithreading capabilities to parallelize computationally intensive tasks, such as residual calculation and forecasting.
- **L-BFGS Optimizer:** The L-BFGS algorithm is used for efficient optimization of the ARIMA model parameters.
- **Module:** Module-based implementation allows for precompilation, driving faster speeds and using vectorization to speed up computations.

## Installation
To use this package, clone the repository and activate the environment:

```sh
git clone https://github.com/yourusername/CustomARIMA.git
cd CustomARIMA
julia --project=. --threads 4
```

Then, install the required dependencies:

```julia
using Pkg
Pkg.instantiate()
```

## Usage

### Fitting an ARIMA Model

```julia
using CustomARIMA

# Example data
y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Fit ARIMA model
model = CustomARIMA.arima(y, 2, 1, 1)

# Print results
println("AR coefficients: ", model.ar_coef)
println("MA coefficients: ", model.ma_coef)
println("Residual variance: ", model.sigma2)
```

### Forecasting

```julia
# Forecast next 12 periods
forecasts = CustomARIMA.forecast(model, 12)
println("Forecasts: ", forecasts)
```

### Visualization
To visualize the forecasting results, you can use the `plot_series` function:

```julia
Y_df = DataFrame(ds=collect(1:length(y)), Passengers=y)
Y_hat_df = DataFrame(ds=collect(length(y)+1:length(y)+12), Passengers=forecasts)
CustomARIMA.plot_series(Y_df, Y_hat_df)
```

## Visualization Example
![Forecasting Visualization](plot.png)
### Diebold Mariano Test
To compare two forecasts to check if one is significantly more accurate than the other
```julia
forecasts1 = [471.92, 491.08, 492.22, 483.94, 474.30, 467.87, 465.62, 466.35, 
              468.27, 470.01, 470.96, 471.14]

forecasts2 = [490.12, 474.28, 469.42, 471.86, 443.67, 433.51, 456.16, 439.73, 
              443.19, 465.33, 449.01, 445.06]
DM_stat, p_value = diebold_mariano_test(y_actual, forecasts1, forecasts2,h=12)
println("DM Statistic: $DM_stat")
println("P-value: $p_value")
```
```
DM Statistic: -0.9459405925081683
P-value: 0.3441788559856751
```
## Performance Benchmarks

- **Execution Time Comparison** - Time reduced by at least 50% relative to the 'Statsforecast' library.
- **Forecast Accuracy Comparison** - <5% in average difference for 10-day forecasts. These results were tested over multiple datasets and hyperparameters.

## Acknowledgments
Inspired by the statsforecast library in Python.
