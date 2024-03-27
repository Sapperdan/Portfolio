

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()

df = pd.read_excel('C:/Users/danam/OneDrive/Desktop/Work/SpaceX.xlsx')

print(df.head(15))


df.columns = df.iloc[0]  # Set the first row as the column names
df = df[1:]  # Remove the first row from the DataFrame


print(df.head())

# Group by "ORIGIN COUNTRY" and "DESTINATION CITY" and count occurrences
route_counts = df.groupby(['ORIGIN COUNTRY', 'DESTINATION CITY']).size()

most_recurring_route = route_counts.idxmax()
most_recurring_route_count = route_counts.max()

print(f"Most Recurring Route: {most_recurring_route} with {most_recurring_route_count} occurrences")


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sb

# Adjusting the group by to consider origin and destination cities
city_route_counts = df.groupby(['ORIGIN CITY', 'DESTINATION CITY']).size().reset_index(name='Count')
city_route_counts['Route'] = city_route_counts['ORIGIN CITY'] + " to " + city_route_counts['DESTINATION CITY']

# Sort the routes by count and select the top N for clearer visualization
top_city_routes = city_route_counts.sort_values(by='Count', ascending=False).head(20)

# Plotting
plt.figure(figsize=(10, 8))
sb.barplot(x='Count', y='Route', data=top_city_routes, palette='viridis')
plt.title('Top Recurring City-to-City Routes')
plt.xlabel('Frequency')
plt.ylabel('Route')
plt.tight_layout()

# Show the plot
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sb

import matplotlib.pyplot as plt
import seaborn as sb

# Convert the route counts to a DataFrame for easy plotting
route_counts_df = route_counts.reset_index(name='Count')
route_counts_df['Route'] = route_counts_df['ORIGIN COUNTRY'] + " to " + route_counts_df['DESTINATION CITY']

# Sort the routes by count and select the top N for clearer visualization
top_routes = route_counts_df.sort_values(by='Count', ascending=False).head(10)

# Plotting
plt.figure(figsize=(10, 8))
sb.barplot(x='Count', y='Route', data=top_routes, palette='viridis')
plt.title('Top Recurring Routes')
plt.xlabel('Frequency')
plt.ylabel('Route')
plt.tight_layout()


plt.show()

----------------------------------------------------------------

# Constants
COST_PER_KG = 5  # Hypothetical average cost per kg

# Monthly spend by lane (in $)
spend = {
    'China': 20e6 * 0.50,
    'Taiwan': 20e6 * 0.24,
    'The Netherlands': 4.5e6 * 0.20,
    'Vietnam': 20e6 * 0.06
}

# Convert spend to weight in kg (assuming $5/kg)
weight_kg = {country: spend[country] / COST_PER_KG for country in spend}

# Convert kg to tons
weight_tons = {country: weight_kg[country] / 1000 for country in weight_kg}

# Create a DataFrame
df_volume = pd.DataFrame(list(weight_tons.items()), columns=['Country', 'Estimated Volume (Tons)'])

# Plotting
df_volume.plot(kind='bar', x='Country', y='Estimated Volume (Tons)', legend=False)
plt.ylabel('Volume in Tons')
plt.title('Estimated Monthly Volume from Country Origin to US')
plt.tight_layout()

# Show the plot
plt.show()

# Return DataFrame for review
df_volume


---------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Hypothetical average cost per kg (same for all lanes for simplicity)
COST_PER_KG = 5

# Monthly spend per country (from the provided spend percentages)
monthly_spend_by_country = {
    'China': 20e6 * 0.50,
    'Taiwan': 20e6 * 0.24,
    'The Netherlands': 4.5e6 * 0.20,
    'Vietnam': 20e6 * 0.06
}

# Estimated monthly spend per city pair (assuming one major city pair per country)
# This is a simplification and might not reflect the actual distribution of spend across different cities
city_pairs = {
    'Shanghai to Los Angeles': monthly_spend_by_country['China'],
    'Taipei to San Francisco': monthly_spend_by_country['Taiwan'],
    'New York to Amsterdam': monthly_spend_by_country['The Netherlands'],
    'Ho Chi Minh City to Seattle': monthly_spend_by_country['Vietnam']
}

# Convert the spend to estimated volume in kg, then to tons
estimated_volume_tons = {city_pair: (spend / COST_PER_KG) / 1000 for city_pair, spend in city_pairs.items()}

# Create a DataFrame for visualization
df_city_volumes = pd.DataFrame(list(estimated_volume_tons.items()), columns=['City Pair', 'Estimated Volume (Tons)'])

# Plotting the estimated volumes
plt.figure(figsize=(10, 6))
df_city_volumes.plot(kind='bar', x='City Pair', y='Estimated Volume (Tons)', legend=False, color='skyblue')
plt.ylabel('Volume in Tons')
plt.title('Estimated Monthly Volume by City Pair')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# Return DataFrame for review
df_city_volumes


# In[23]:


# Monthly spend allocations from previous calculations
spend_china = 20e6 * 0.50
spend_taiwan = 20e6 * 0.24
spend_netherlands = 4.5e6 * 0.20
spend_vietnam = 20e6 * 0.06

# Sum of monthly spends
total_monthly_spend = spend_china + spend_taiwan + spend_netherlands + spend_vietnam

# Estimated weights in kg for each country (from previous calculations)
weight_kg_china = spend_china / COST_PER_KG
weight_kg_taiwan = spend_taiwan / COST_PER_KG
weight_kg_netherlands = spend_netherlands / COST_PER_KG
weight_kg_vietnam = spend_vietnam / COST_PER_KG

# Total estimated weight in kg
total_estimated_weight_kg = weight_kg_china + weight_kg_taiwan + weight_kg_netherlands + weight_kg_vietnam

# Average price per kilo
average_price_per_kilo = total_monthly_spend / total_estimated_weight_kg
average_price_per_kilo



# Re-calculate the estimated weights in kg for each country using the tonnage provided earlier
weight_kg_china = 2000.0 * 1000  # from tons to kg
weight_kg_taiwan = 960.0 * 1000  # from tons to kg
weight_kg_netherlands = 180.0 * 1000  # from tons to kg
weight_kg_vietnam = 240.0 * 1000  # from tons to kg

# Calculate the total estimated weight in kg for all lanes combined
total_estimated_weight_kg = weight_kg_china + weight_kg_taiwan + weight_kg_netherlands + weight_kg_vietnam

# Calculate the average price per kilo for all lanes
average_price_per_kilo = total_monthly_spend / total_estimated_weight_kg

average_price_per_kilo


# Let's first convert the estimated volume in tons back to kg
estimated_volume_kg = {city_pair: tons * 1000 for city_pair, tons in estimated_volume_tons.items()}

# Now calculate the average price per kilo for each city pair
average_price_per_kilo = {city_pair: spend / estimated_volume_kg[city_pair] 
                          for city_pair, spend in city_pairs.items()}

# Calculate the total volume in kg and total spend for the weighted average calculation
total_volume_kg = sum(estimated_volume_kg.values())
total_spend = sum(city_pairs.values())

# Calculate the weighted average price per kilo across all city pairs
weighted_average_price_per_kilo = total_spend / total_volume_kg

# Let's also create a DataFrame for the average price per kilo for each city pair for display purposes
df_price_per_kilo = pd.DataFrame(list(average_price_per_kilo.items()), 
                                 columns=['City Pair', 'Average Price per Kilo ($)'])

weighted_average_price_per_kilo, df_price_per_kilo




estimated_volume_kg = {city_pair: tons * 1000 for city_pair, tons in estimated_volume_tons.items()}

# Now calculate the average price per kilo for each city pair
average_price_per_kilo = {city_pair: spend / estimated_volume_kg[city_pair] 
                          for city_pair, spend in city_pairs.items()}

# Calculate the total volume in kg and total spend for the weighted average calculation
total_volume_kg = sum(estimated_volume_kg.values())
total_spend = sum(city_pairs.values())

# Calculate the weighted average price per kilo across all city pairs
weighted_average_price_per_kilo = total_spend / total_volume_kg

# Let's also create a DataFrame for the average price per kilo for each city pair for display purposes
df_price_per_kilo = pd.DataFrame(list(average_price_per_kilo.items()), 
                                 columns=['City Pair', 'Average Price per Kilo ($)'])

weighted_average_price_per_kilo, df_price_per_kilo


plt.figure(figsize=(10, 6))
bar_plot = plt.bar(df_price_per_kilo['City Pair'], df_price_per_kilo['Average Price per Kilo ($)'], color='lightcoral')
plt.ylabel('Average Price per Kilo ($)')
plt.title('Average Price per Kilo by City Pair')
plt.xticks(rotation=45)
plt.tight_layout()

# Adding the values on top of the bars
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()


# Existing estimated volume in tons
existing_lanes = {
    'Shanghai to Los Angeles': 2000.0,
    'Taipei to San Francisco': 960.0,
    'New York to Amsterdam': 180.0,
    'Ho Chi Minh City to Seattle': 240.0
}

# Hypothetical additional lanes with assumed smaller volumes
additional_lanes = {
    'Beijing to New York': 150.0,
    'Hong Kong to Chicago': 130.0,
    'Seoul to Dallas': 110.0,
    'Singapore to Houston': 90.0,
    'Bangkok to Miami': 70.0,
    'Kuala Lumpur to Atlanta': 50.0
}

# Combine the existing lanes with the hypothetical additional lanes
all_lanes = {**existing_lanes, **additional_lanes}

# Create DataFrame for visualization
df_all_lanes = pd.DataFrame(list(all_lanes.items()), columns=['City Pair', 'Estimated Volume (Tons)'])

# Plotting the estimated volumes for all 10 lanes
plt.figure(figsize=(12, 8))
bar_plot = plt.bar(df_all_lanes['City Pair'], df_all_lanes['Estimated Volume (Tons)'], color='skyblue')
plt.ylabel('Estimated Volume in Tons')
plt.title('Estimated Monthly Volume by City Pair')
plt.xticks(rotation=60)
plt.tight_layout()

# Adding the values on top of the bars
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()

# Return DataFrame for review
df_all_lanes


# In[29]:


# Combine the existing lanes with the hypothetical additional lanes
all_lanes = {**existing_lanes, **additional_lanes}

# Sort the lanes by volume to get the top ones, and if we have less than 10, we keep the original number
top_lanes_by_volume = dict(sorted(all_lanes.items(), key=lambda item: item[1], reverse=True)[:10])

# Create DataFrame for visualization
df_top_lanes = pd.DataFrame(list(top_lanes_by_volume.items()), columns=['City Pair', 'Estimated Volume (Tons)'])

# Plotting the estimated volumes for the top lanes
plt.figure(figsize=(12, 8))
bar_plot = plt.bar(df_top_lanes['City Pair'], df_top_lanes['Estimated Volume (Tons)'], color='skyblue')
plt.ylabel('Estimated Volume in Tons')
plt.title('Top Estimated Monthly Volumes by City Pair')
plt.xticks(rotation=60)
plt.tight_layout()

# Adding the values on top of the bars
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()

# Return DataFrame for review
df_top_lanes


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt

# Let's redefine the existing lanes and hypothetical additional lanes due to the reset of the code execution state
existing_lanes = {
    'Shanghai to Los Angeles': 2000.0,
    'Taipei to San Francisco': 960.0,
    'New York to Amsterdam': 180.0,
    'Ho Chi Minh City to Seattle': 240.0
}

# Hypothetical additional lanes with assumed smaller volumes
additional_lanes = {
    'Beijing to New York': 150.0,
    'Hong Kong to Chicago': 130.0,
    'Seoul to Dallas': 110.0,
    'Singapore to Houston': 90.0,
    'Bangkok to Miami': 70.0,
    'Kuala Lumpur to Atlanta': 50.0
}

# Combine the existing lanes with the hypothetical additional lanes
all_lanes = {**existing_lanes, **additional_lanes}

# Create DataFrame for visualization
df_all_lanes = pd.DataFrame(list(all_lanes.items()), columns=['City Pair', 'Estimated Volume (Tons)'])

# Plotting the estimated volumes for all 10 lanes
plt.figure(figsize=(12, 8))
bar_plot = plt.bar(df_all_lanes['City Pair'], df_all_lanes['Estimated Volume (Tons)'], color='skyblue')
plt.ylabel('Estimated Volume in Tons')
plt.title('Estimated Monthly Volume by City Pair')
plt.xticks(rotation=60)
plt.tight_layout()

# Adding the values on top of the bars
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()

# Return DataFrame for review
df_all_lanes


# In[31]:


# Re-import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Hypothetical average cost per kg (same for all lanes for simplicity)
COST_PER_KG = 5

# Monthly spend per country (from the provided spend percentages)
monthly_spend_by_country = {
    'China': 20e6 * 0.50,
    'Taiwan': 20e6 * 0.24,
    'The Netherlands': 4.5e6 * 0.20,
    'Vietnam': 20e6 * 0.06
}

# Existing lanes with estimated volume in tons (from previous calculations)
existing_lanes = {
    'Shanghai to Los Angeles': 2000.0,
    'Taipei to San Francisco': 960.0,
    'New York to Amsterdam': 180.0,
    'Ho Chi Minh City to Seattle': 240.0
}

# Assuming 6 additional hypothetical lanes, distributing remaining spend equally among them
# First, calculate the remaining spend that is not allocated to the existing lanes
total_spend = sum(monthly_spend_by_country.values())
allocated_spend = sum(existing_lanes.values()) * COST_PER_KG  # Convert tons back to spend
remaining_spend = total_spend - allocated_spend

# Hypothetical additional lanes with assumed volumes based on remaining spend
additional_lanes = {
    'Beijing to New York': remaining_spend / 6 / COST_PER_KG / 1000,
    'Hong Kong to Chicago': remaining_spend / 6 / COST_PER_KG / 1000,
    'Seoul to Dallas': remaining_spend / 6 / COST_PER_KG / 1000,
    'Singapore to Houston': remaining_spend / 6 / COST_PER_KG / 1000,
    'Bangkok to Miami': remaining_spend / 6 / COST_PER_KG / 1000,
    'Kuala Lumpur to Atlanta': remaining_spend / 6 / COST_PER_KG / 1000
}

# Combine the existing lanes with the hypothetical additional lanes
all_lanes_volumes = {**existing_lanes, **additional_lanes}

# Create DataFrame for visualization
df_all_lanes_volumes = pd.DataFrame(list(all_lanes_volumes.items()), columns=['City Pair', 'Estimated Volume (Tons)'])

# Plotting the estimated volumes for all 10 lanes
plt.figure(figsize=(12, 8))
bar_plot = plt.bar(df_all_lanes_volumes['City Pair'], df_all_lanes_volumes['Estimated Volume (Tons)'], color='skyblue')
plt.ylabel('Estimated Volume in Tons')
plt.title('Estimated Monthly Volume by City Pair')
plt.xticks(rotation=60)
plt.tight_layout()

# Adding the values on top of the bars
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()

# Return DataFrame for review
df_all_lanes_volumes


# In[32]:


# To add the country origin to each city pair, we'll manually map each city to its respective country.
# This will help clarify where the cargo is originating from for each lane.

country_origin_map = {
    'Shanghai to Los Angeles': 'China to USA',
    'Taipei to San Francisco': 'Taiwan to USA',
    'New York to Amsterdam': 'USA to Netherlands',
    'Ho Chi Minh City to Seattle': 'Vietnam to USA',
    'Beijing to New York': 'China to USA',
    'Hong Kong to Chicago': 'China to USA',
    'Seoul to Dallas': 'South Korea to USA',
    'Singapore to Houston': 'Singapore to USA',
    'Bangkok to Miami': 'Thailand to USA',
    'Kuala Lumpur to Atlanta': 'Malaysia to USA'
}

# Update the 'City Pair' column in df_all_lanes_volumes to include the country origin
df_all_lanes_volumes['City Pair with Country Origin'] = df_all_lanes_volumes['City Pair'].map(country_origin_map)

# Plotting the estimated volumes with country origin included in the city pairs
plt.figure(figsize=(14, 10))
bar_plot = plt.bar(df_all_lanes_volumes['City Pair with Country Origin'], df_all_lanes_volumes['Estimated Volume (Tons)'], color='skyblue')
plt.ylabel('Estimated Volume in Tons')
plt.title('Estimated Monthly Volume by City Pair with Country Origin')
plt.xticks(rotation=90)
plt.tight_layout()

# Adding the values on top of the bars
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()

# Return updated DataFrame for review
df_all_lanes_volumes[['City Pair with Country Origin', 'Estimated Volume (Tons)']]


# In[23]:


# To add the country origin to each city pair, we'll manually map each city to its respective country.
# This will help clarify where the cargo is originating from for each lane.

country_origin_map = {
    'Shanghai to Los Angeles': 'China to USA',
    'Taipei to San Francisco': 'Taiwan to USA',
    'New York to Amsterdam': 'USA to Netherlands',
    'Ho Chi Minh City to Seattle': 'Vietnam to USA',
    'Beijing to New York': 'China to USA',
    'Hong Kong to Chicago': 'China to USA',
    'Seoul to Dallas': 'South Korea to USA',
    'Singapore to Houston': 'Singapore to USA',
    'Bangkok to Miami': 'Thailand to USA',
    'Kuala Lumpur to Atlanta': 'Malaysia to USA'
}

# Update the 'City Pair' column in df_all_lanes_volumes to include the country origin
df_all_lanes_volumes['City Pair with Country Origin'] = df_all_lanes_volumes['City Pair'].map(country_origin_map)

# Plotting the estimated volumes with country origin included in the city pairs
plt.figure(figsize=(14, 10))
bar_plot = plt.bar(df_all_lanes_volumes['City Pair with Country Origin'], df_all_lanes_volumes['Estimated Volume (Tons)'], color='skyblue')
plt.ylabel('Estimated Volume in Tons')
plt.title('Estimated Monthly Volume by City Pair with Country Origin')
plt.xticks(rotation=90)
plt.tight_layout()

# Adding the values on top of the bars
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()

# Return updated DataFrame for review
df_all_lanes_volumes[['City Pair with Country Origin', 'Estimated Volume (Tons)']]


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Assuming df is your DataFrame with 'Lane ID' and 'Volume'
# Encode 'Lane ID' using one-hot encoding
encoder = OneHotEncoder(sparse=False)
lane_id_encoded = encoder.fit_transform(df[['LANE ID']])

# Create a new DataFrame with encoded lane IDs
lane_id_df = pd.DataFrame(lane_id_encoded, columns=encoder.get_feature_names_out(['Lane ID']))

# Concatenate the original DataFrame (minus the 'Lane ID' column) with the new encoded DataFrame
df_encoded = pd.concat([df.drop('Lane ID', axis=1), lane_id_df], axis=1)


# Define features (X) and target (y)
X = df_encoded.drop('Volume', axis=1)
y = df['Volume']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")



# In[37]:


# Assuming df is your DataFrame
from sklearn.compose import ColumnTransformer

# Initialize OneHotEncoder within a ColumnTransformer to handle categorical encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('LANE ID', OneHotEncoder(sparse=False), ['LANE ID'])
    ],
    remainder='drop'  # Drop other columns not specified in transformers
)

# Fit and transform the training data
X_transformed = preprocessor.fit_transform(df[['LANE ID']])
feature_names = preprocessor.named_transformers_['LANE ID'].get_feature_names_out()

# Create a new DataFrame for the transformed features
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

# Concatenate with the Volume column if it's dropped by 'remainder' option
df_transformed = pd.concat([X_transformed_df, df[['Volume']].reset_index(drop=True)], axis=1)

X = df_transformed.drop('Volume', axis=1)
y = df_transformed['Volume']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")



# In[2]:


print(df.columns)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




