
import math
import scipy as sp
import numpy as np


import pandas as pd



import matplotlib.pyplot as plt

from scipy import stats
import seaborn as sns
sns.set()


import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


import warnings
from statsmodels.tsa.seasonal import seasonal_decompose # Error Trend Seasonality decomposition


from statsmodels.tsa.holtwinters import ExponentialSmoothing # double and triple exponential smoothing

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
warnings.filterwarnings('ignore')
series = pd.read_excel( 'random_time_series.xlsx', squeeze=True) 
series['Doc_Date'] = pd.to_datetime(series['Doc_Date'], format='%d.%m.%Y')
series.set_index('Doc_Date', inplace=True)
    
series= series.resample('MS').sum()

series.astype('float')
series['Transport_cost']=np.floor(series['Transport_cost'].astype(float)+0.5)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
series.to_excel('random_time_series_month_resampl.xlsx', sheet_name='Sheet_name_1')
print(series.shape)


 
# transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(series.squeeze())
 

fig, ax = plt.subplots(1, 2)
 
sns.distplot(series, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Non-Normal", color ="red", ax = ax[0])
 
sns.distplot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Normal", color ="green", ax = ax[1])
 

plt.legend(loc = "upper right")
 
fig.set_figheight(5)
fig.set_figwidth(10)
 
print(f"Lambda value used for Transformation: {fitted_lambda}")

# plots for standard distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
sns.histplot(series,kde=True, color ='blue',ax=ax[0])
sm.ProbPlot(series).qqplot(line='s', ax=ax[1])

analysis = fitted_data.copy()


decompose_result_mult = seasonal_decompose(analysis, model='add', period =4)

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

decompose_result_mult.plot()
plt.show()



# Split into train and test set
train = series[:48] 
fitted_data1, fitted_lambda1 = stats.boxcox(train.squeeze())
print(f"Lambda value used for Transformation: {fitted_lambda1}")
test = series[48:]



fitted_model = ExponentialSmoothing(train['Transport_cost'],trend = 'mul',seasonal='add',seasonal_periods=4,use_boxcox = True) 
fitted1 = fitted_model.fit(optimized = True, remove_bias=False)


print(fitted1.summary())
test_predictions = fitted1.forecast(9).rename('HW Test Forecast')
pd.set_option('display.max_rows', 1500)
print(test_predictions[:9])



train['Transport_cost'].plot(legend=True,label='TRAIN')
test['Transport_cost'].plot(legend=True,label='TEST',figsize=(12,8))
plt.title('Train and Test Data')
plt.show()
train['Transport_cost'].plot(legend=True,label='TRAIN')
test['Transport_cost'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters')

plt.show()


from sklearn.metrics import mean_absolute_error,mean_squared_error
mse =mean_squared_error(test,test_predictions)
print(f'Mean Absolute Error = {mean_absolute_error(test,test_predictions)}')
print(f'Mean Squared Error = {mean_squared_error(test,test_predictions)}')
rmse = math.sqrt(mse)
print(rmse)
print(test.describe())

#final model for Exponential Smoothing method #1 
final_model = ExponentialSmoothing(series['Transport_cost'],trend = 'mul', seasonal='add',seasonal_periods=4,use_boxcox = True )
fitted = final_model.fit( optimized = True, remove_bias=False)

print(fitted.summary())
forecast_predictions = fitted.forecast(steps=15)

series['Transport_cost'].plot(figsize=(12,8),legend=True,label='Current costs')
forecast_predictions.plot(legend=True,label='Forecasted costs')
plt.title('Transport Forecast')
plt.show()
pd.set_option('display.max_rows', 1000)
print(forecast_predictions[:15])


print(series.describe())

from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Build model.
ets_model = ETSModel(
    endog=train['Transport_cost'], 
    trend = 'mul',
    seasonal='add',
    error = 'add',
    seasonal_periods=4
    
    
)
ets_result = ets_model.fit()




print(ets_result.summary())
test_predictions1 = ets_result.forecast(9).rename('HW Test Forecast')
pd.set_option('display.max_rows', 1500)
print(test_predictions1[:9])
print(test.describe())
mse1 =mean_squared_error(test,test_predictions1)
print(f'Mean Absolute Error = {mean_absolute_error(test,test_predictions1)}')
print(f'Mean Squared Error = {mean_squared_error(test,test_predictions1)}')
rmse1 = math.sqrt(mse1)
print(rmse1)
train['Transport_cost'].plot(figsize=(12,8),legend=True,label='Current costs')
test['Transport_cost'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions1.plot(legend=True,label='Forecasted costs')
plt.title('Transport Forecast')
plt.show()
pd.set_option('display.max_rows', 1000)
print(test_predictions1[:9])

#final model for ETS method #2
final_model1 =ETSModel(series['Transport_cost'],trend='mul', seasonal='add', error = 'add',seasonal_periods=4)
fitted_res = final_model1.fit()

print(fitted_res.summary())
forecast_predictions1 = fitted_res.forecast(steps=15)

series['Transport_cost'].plot(figsize=(12,8),legend=True,label='Current costs')
forecast_predictions1.plot(legend=True,label='Forecasted costs')
plt.title('Transport Forecast')
plt.show()
pd.set_option('display.max_rows', 1000)
print(forecast_predictions1[:15])


pred = fitted_res.get_prediction(start='2018-01-01 00:00:00', end='2022-10-01 00:00:00')
df = pred.summary_frame(alpha=0.05)
print(df)