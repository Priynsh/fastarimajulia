module CustomARIMA

using Optim, LinearAlgebra, Statistics

export ARIMAModel, arima, forecast

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
end