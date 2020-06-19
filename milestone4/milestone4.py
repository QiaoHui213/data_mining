# Import module
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
import seaborn

# Read Data
df = pd.read_csv("C:\\Users\\User\\Desktop\\data_mining\\data_mining\\goldprice_12-Mar-2020_22-26-27.csv")
print(df.head(10))

# Change the format of "Date"
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d')
# Remove the "," in the "Price", "Open', 'Low', and 'High' column and change the "string" type to "float"
df['Price']=df['Price'].astype(str).str.replace(',', '').astype(float)
df['Open']=df['High'].astype(str).str.replace(',', '').astype(float)
df['Low']=df['Low'].astype(str).str.replace(',', '').astype(float)
df['High']=df['High'].astype(str).str.replace(',', '').astype(float)
df['Volume']=df['Volume'].replace({'K': '*1e3', '-': '1'}, regex=True).map(pd.eval)
df['Change %']=df['Change %'].replace({'%': '*1e-2'}, regex=True).map(pd.eval)
df['Volume'] = df['Volume'].replace(1.0,np.NaN)

# Select Date and Price only
df_price = df[['Date', 'Price']]
df_price.dropna()
print(df_price.head(10))
# Check if there contain null in the attributes
print(df_price.isnull().any())

# Set up the function of selecting range of date and commodity type
def extract_data(start_date,end_date,commodity_type):
  if commodity_type == "Gold":
    return df_price[(df_price.Date>=start_date)&(df_price.Date<=end_date)]
  elif commodity_type == "Silver":
    return df_price[(df_price.Date>=start_date)&(df_price.Date<=end_date)]

# Specify users inputs, you can change the start date and end data to extract data from G-2019
# In this case, I select the data from 2019-01-01 to 2019-11-12 and gold as my input
start_date = "20190101"
end_date = "20191112"
commodity_type = "Trends of Gold Price"
df_gold= extract_data(start_date, end_date, commodity_type)

# Change "Date" as index and sort the data
df_price['Date'] =pd.to_datetime(df_price.Date)
df_sort = df_price.sort_values('Date')

print(df_sort.head())
print(df_sort.describe())

# Visualize the Data
df_sort.plot(x='Date',y='Price')
plt.title(commodity_type)
plt.show()

df_sort['S_3'] = df_sort['Price'].shift(1).rolling(window=3).mean()
df_sort['S_9']= df_sort['Price'].shift(1).rolling(window=9).mean()
df_sort= df_sort.dropna()
X = df_sort[['S_3','S_9']]
print(X.head())

y = df_sort['Price']
print(y.head())

t=.8
t = int(t*len(df_sort))
# Train dataset
X_train = X[:t]
y_train = y[:t]
# Test dataset
X_test = X[t:]
y_test = y[t:]

# Linear Regression
linear = LinearRegression().fit(X_train,y_train)
print ("Gold Price =", round(linear.coef_[0],2), "* 3 Days Moving Average", round(linear.coef_[1],2), "* 9 Days Moving Average +", round(linear.intercept_,2))

predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
predicted_price.plot(figsize=(10,5))
y_test.plot()
plt.legend(['predicted_price','actual_price'])
plt.ylabel("Gold Price")
plt.show()

r2_score = linear.score(X[t:],y[t:])*100
float("{0:.2f}".format(r2_score))