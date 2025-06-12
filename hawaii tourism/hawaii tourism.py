import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the Excel file
df_2022 = pd.read_excel("2022-oahu.xlsx")
df_2023 = pd.read_excel("2023-oahu.xlsx")

# Grab row 2 from both
row_2022 = df_2022.loc[2, ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]

row_2023 = df_2023.loc[2, ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]



# Convert to DataFrames
data_2022 = pd.DataFrame({
    'Date': pd.date_range(start='2022-01-01', periods=12, freq='MS'),
    'Visitor_Arrivals': row_2022.values
})

data_2023 = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=12, freq='MS'),
    'Visitor_Arrivals': row_2023.values
})

# Combine both years
full_data = pd.concat([data_2022, data_2023], ignore_index=True)

# Make sure numeric
full_data['Visitor_Arrivals'] = pd.to_numeric(full_data['Visitor_Arrivals'])

# Rename for Prophet
prophet_df = full_data.rename(columns={'Date': 'ds', 'Visitor_Arrivals': 'y'})

# fitting prophet
model = Prophet()
model.fit(prophet_df)

# Forecast 6 future months
future = model.make_future_dataframe(periods=6, freq='MS')
forecast = model.predict(future)

# Plot it


fig = model.plot(forecast)
plt.title("Forecast of Visitor Arrivals to Oʻahu (with 2022–2023 data)")
plt.xlabel("Date")
plt.ylabel("Visitor Arrivals")
plt.show()
#
#
# # Make future dates
# future = model.make_future_dataframe(periods=6, freq='MS')
#
# # Predict future values
# forecast = model.predict(future)
#
#
# fig = model.plot(forecast)
# plt.title("Forecast of Visitor Arrivals to Oʻahu")
# plt.xlabel("Date")
# plt.ylabel("Visitor Arrivals")
# plt.show()

# Show the cleaned dataset


# plt.plot(cleaned['Date'], cleaned['Visitor_Arrivals'])
# plt.title("Original Visitor Data")
# plt.xlabel("Date")
# plt.ylabel("Visitor Arrivals")
#
# # Set x-axis to show every month
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()






