import timeit

def run_arima():
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import ARIMA

    df = pd.read_csv('airline_passengers.csv')
    df['ds'] = pd.to_datetime(df['Month'])
    df['y'] = df['Passengers']
    df['unique_id'] = 1
    df = df[['unique_id', 'ds', 'y']]

    p, d, q = 2, 1, 1
    sf = StatsForecast(
        models=[ARIMA(order=(p, d, q))],
        freq='ME'
    )
    sf.fit(df)
    forecast = sf.predict(h=12)
    return forecast

execution_time = timeit.timeit(run_arima, number=1)
print(f"Execution time: {execution_time:.4f} seconds")