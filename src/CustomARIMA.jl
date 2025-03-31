module CustomARIMA
using Plots, DataFrames, Dates
using Optim, LinearAlgebra, Statistics
export ARIMAModel, arima, forecast, plot_series,forecast_rolling,diebold_mariano_test

struct ARIMAModel
    p::Int
    d::Int
    q::Int
    ar_coef::Vector{Float64}
    ma_coef::Vector{Float64}
    sigma2::Float64
    residuals::Vector{Float64}
    y_orig::Vector{Float64}
end
function difference(y::AbstractVector, d::Int)
    for _ in 1:d
        y = diff(y)
    end
    return y
end
const log2π = log(2π)
function arima(y::AbstractVector{<:Real}, p::Int, d::Int, q::Int; maxiter=1000)
    y_orig = float.(y)
    y_diff = difference(y_orig, d)
    n = length(y_diff)
    function obj(θ)
        ar = θ[1:p]
        ma = θ[p+1:p+q]
        log_sigma2 = θ[end]
        sigma2 = exp(log_sigma2)
        residuals = zeros(n)
        for t in max(p, q)+1:n
            ar_term = 0.0
            for i in 1:p
                (t - i) < 1 && break
                ar_term += ar[i] * y_diff[t-i]
            end
            ma_term = 0.0
            for j in 1:q
                (t - j) < 1 && break
                ma_term += ma[j] * residuals[t-j]
            end
            residuals[t] = y_diff[t] - ar_term - ma_term
        end
        log_term = -n/2 * (log2π + log_sigma2)
        sum_sq = sum(residuals.^2)
        -(log_term - sum_sq / (2 * sigma2))
    end
    θ0 = vcat(zeros(p), zeros(q), [log(var(y_diff))])
    result = optimize(obj, θ0, LBFGS(), Optim.Options(
        iterations=maxiter,
        x_tol=1e-4,
        g_tol=1e-4
    ))
    ar = result.minimizer[1:p]
    ma = result.minimizer[p+1:p+q]
    sigma2 = exp(result.minimizer[end])
    final_residuals = zeros(n)
    let θ = result.minimizer
        ar = θ[1:p]
        ma = θ[p+1:p+q]
        for t in max(p, q)+1:n
            ar_term = sum(ar[i] * y_diff[t-i] for i in 1:min(p, t-1))
            ma_term = sum(ma[j] * final_residuals[t-j] for j in 1:min(q, t-1))
            final_residuals[t] = y_diff[t] - ar_term - ma_term
        end
    end
    ARIMAModel(p, d, q, ar, ma, sigma2, final_residuals, y_orig)
end


function plot_series(Y_df::DataFrame, Y_hat_df::DataFrame)
    Y_df.ds = Date.(Y_df.ds)
    Y_hat_df.ds = Date.(Y_hat_df.ds)

    actual_colors = [:blue, :green, :red, :purple, :orange, :cyan, :magenta, :black, :pink]
    forecast_colors = [:dodgerblue, :limegreen, :firebrick, :indigo, :darkorange, :teal, :deeppink, :gray, :lightcoral]

    plt = plot(title="Time Series Forecasting", xlabel="Date", ylabel="Value", legend=:topleft, xtickfont=8, xrotation=45)

    for (i, col) in enumerate(names(Y_df)[2:end])
        color = actual_colors[mod1(i, length(actual_colors))]
        plot!(plt, Y_df.ds, Y_df[!, col], label="Actual - $col", color=color, linewidth=2)
    end

    for (i, col) in enumerate(names(Y_hat_df)[2:end])
        color = forecast_colors[mod1(i, length(forecast_colors))]
        plot!(plt, Y_hat_df.ds, Y_hat_df[!, col], label="Forecast - $col", color=color, linewidth=2)
    end

    display(plt)
end
function normal_cdf(x::T) where T<:AbstractFloat
    t = one(T) / (one(T) + T(0.2316419) * abs(x))
    b = T[0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    poly = t * (b[1] + t * (b[2] + t * (b[3] + t * (b[4] + t * b[5]))))
    cdf = one(T) - (inv(sqrt(T(2π))) * exp(-T(0.5) * x^2) * poly)
    return x < zero(T) ? one(T) - cdf : cdf
end

function diebold_mariano_test(actual::Vector{T}, forecast1::Vector{T}, forecast2::Vector{T}; h::Int=1) where T<:Number
    T_size = length(actual)
    @assert T_size == length(forecast1) == length(forecast2) "All vectors must be the same length"
    e1 = actual .- forecast1
    e2 = actual .- forecast2
    d_t = e1 .^ 2 .- e2 .^ 2
    mean_d = sum(d_t) / T_size
    var_d = sum((d_t .- mean_d) .^ 2) / (T_size - 1)
    gamma_h = zero(T)
    for h in 1:min(h, T_size - 1)
        sum_h = sum((d_t[t] - mean_d) * (d_t[t-h] - mean_d) for t in h+1:T_size)
        gamma_h += 2 * (one(T) - T(h) / (h+1)) * (sum_h / T_size)
    end
    S_d = var_d + gamma_h
    DM_stat = mean_d / sqrt(S_d / T_size)
    p_value = 2 * (one(T) - normal_cdf(abs(DM_stat)))
    return DM_stat, p_value
end

function forecast(model::ARIMAModel, steps::Int)
    y_diff = model.residuals
    p, q = model.p, model.q
    ar = model.ar_coef
    ma = model.ma_coef
    forecasts = zeros(steps)
    total_length = length(y_diff) + steps
    hist_diff = Vector{Float64}(undef, total_length)
    hist_resid = Vector{Float64}(undef, total_length)
    hist_diff[1:length(y_diff)] .= y_diff
    hist_resid[1:length(y_diff)] .= model.residuals
    for i in 1:steps
        current_idx = length(y_diff) + i
        ar_term = sum(ar[j] * hist_diff[current_idx-j] for j in 1:p if current_idx-j > 0)
        ma_term = sum(ma[j] * hist_resid[current_idx-j] for j in 1:q if current_idx-j > 0)
        new_val = ar_term + ma_term
        hist_diff[current_idx] = new_val
        hist_resid[current_idx] = 0.0
        forecasts[i] = new_val
    end
    for _ in 1:model.d
        forecasts = cumsum(vcat(model.y_orig[end], forecasts))[2:end]
    end 
    forecasts
end
function forecast_rolling(model::ARIMAModel, steps::Int, window_size::Int)
    y_diff = model.residuals
    p, q = model.p, model.q
    ar = model.ar_coef
    ma = model.ma_coef
    forecasts = zeros(steps)
    hist_diff = copy(y_diff[max(1, end - window_size + 1):end])
    hist_resid = copy(model.residuals[max(1, end - window_size + 1):end])

    for i in 1:steps
        ar_term = sum(ar[j] * hist_diff[end-j+1] for j in 1:p if j <= length(hist_diff))
        ma_term = sum(ma[j] * hist_resid[end-j+1] for j in 1:q if j <= length(hist_resid))
        new_val = ar_term + ma_term
        if length(hist_diff) >= window_size
            popfirst!(hist_diff) 
        end
        if length(hist_resid) >= window_size
            popfirst!(hist_resid) 
        end
        push!(hist_diff, new_val)
        push!(hist_resid, 0.0)

        forecasts[i] = new_val
    end
    for _ in 1:model.d
        forecasts = cumsum(vcat(model.y_orig[end], forecasts))[2:end]
    end 
    forecasts
end

end