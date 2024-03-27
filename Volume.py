

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()


# Environment setup
df = pd.read_excel('C:/Users/danam/OneDrive/Desktop/Work/SpaceX.xlsx')
data = pd.read_excel('C:/Users/danam/OneDrive/Desktop/Work/SpaceX.xlsx')

print(df.head())


# In[13]:


# Monthly spend averages for each transportation mode
monthly_spend = {
    "US Air Import": 20000000,  # $20 million
    "US Air Export": 4500000,   # $4.5 million
    "US Ocean Export": 500000   # $500k
}

# Client's spend percentage breakdown by country for air import/export
country_spend_breakdown = {
    "US Air Import from China": 0.50,
    "US Air Import from Taiwan": 0.24,
    "US Air Export to The Netherlands": 0.20,  # Assuming this is from the US
    "US Air Import from Vietnam": 0.06
}

# Calculate the estimated monthly spend for each country pairing
estimated_monthly_spend_by_country = {
    key: monthly_spend["US Air Import"] * percentage if "Import" in key else
    monthly_spend["US Air Export"] * percentage for key, percentage in country_spend_breakdown.items()
}

estimated_monthly_spend_by_country



# Since the direct mapping by country didn't work, let's attempt to identify lanes by origin and destination cities

country_city_mapping = {
    "China": ["DONGGUAN"],  # Assuming Dongguan represents China for simplicity
    "Taiwan": ["CHUNG LI"],  # Assuming Chung Li represents Taiwan
    "The Netherlands": [],  # No specific cities provided for Netherlands in the export data
    "Vietnam": ["HAIPHONG"]  # Assuming Haiphong represents Vietnam
}

# We'll check the dataset for these cities and infer the lanes based on matches
# For this, we need to identify if any of the origin or destination cities match our mapping

# Checking for any matching origin cities in the dataset
filtered_data_with_cities = data[data['ORIGIN CITY'].isin(sum(country_city_mapping.values(), [])) |
                                 data['DESTINATION CITY'].isin(sum(country_city_mapping.values(), []))]

# Display a summary of the filtered lanes based on city matches
filtered_summary_with_cities = filtered_data_with_cities[['LANE ID', 'ORIGIN CITY', 'DESTINATION CITY']].drop_duplicates()
filtered_summary_with_cities





# We will then distribute this total estimated volume across all lanes based on the number of lanes

# Calculate total estimated kilograms for the dataset
total_kg_estimated = sum(kg_estimates.values())

# Identify all unique origin-destination city pairings in the dataset
all_city_pairs = data[['ORIGIN CITY', 'DESTINATION CITY']].drop_duplicates()

# Assuming an even distribution of volume across all lanes for simplicity
# (This is a simplification and likely does not reflect the true distribution of volumes)
num_lanes = len(all_city_pairs)
estimated_kg_per_lane = total_kg_estimated / num_lanes

# Add estimated volume to each city pairing
all_city_pairs['Estimated Kilograms'] = estimated_kg_per_lane

all_city_pairs.reset_index(drop=True, inplace=True)
all_city_pairs


# In[21]:


import pandas as pd

# Assuming an average cost of $5 per kilogram for simplicity
cost_per_kg = 2.60

# Financial spend breakdown by country
country_spend_breakdown = {
    "US Air Import from China": 10000000.0,  # Monthly spend
    "US Air Import from Taiwan": 4800000.0,
    "US Air Export to The Netherlands": 900000.0,
    "US Air Import from Vietnam": 1200000.0
}

# Calculate the estimated volume in kilograms for each country pairing
kg_estimates = {
    country: spend / cost_per_kg * 12  # Convert monthly spend to annual and then to kilograms
    for country, spend in country_spend_breakdown.items()
}

# Display the estimated kilograms
print("Estimated Kilograms for Each Country Pairing:")
for country, kg in kg_estimates.items():
    print(f"{country}: {kg:.2f} kg")

actual_volumes = kg_estimates  # Hypothetical actual volumes, in this case, our estimates
price_per_kg_estimates = {
    country: (spend * 12) / actual_volumes[country]
    for country, spend in country_spend_breakdown.items()
}

# Display the estimated price per kilogram
print("\nEstimated Price per Kilogram:")
for country, price in price_per_kg_estimates.items():
    print(f"{country}: ${price:.2f} per kg")




unique_country_city_pairs = data[['ORIGIN COUNTRY', 'DESTINATION CITY']].drop_duplicates().reset_index(drop=True)



# In[22]:


# Define the monthly spend and calculate the annual spend for air import and export
monthly_spend = {
    "US Air Import": 20e6,  # $20 million
    "US Air Export": 4.5e6  # $4.5 million
}
annual_spend = {key: value * 12 for key, value in monthly_spend.items()}

# Spend breakdown by country (as a percentage of relevant spend)
spend_breakdown_percentage = {
    "China": 0.50,  # 50% of air import spend
    "Taiwan": 0.24,  # 24% of air import spend
    "Netherlands": 0.20,  # 20% of air export spend
    "Vietnam": 0.06  # 6% of air import spend
}

# Annual spend by route
annual_spend_by_route = {
    "China to USA": annual_spend["US Air Import"] * spend_breakdown_percentage["China"],
    "Taiwan to USA": annual_spend["US Air Import"] * spend_breakdown_percentage["Taiwan"],
    "USA to Netherlands": annual_spend["US Air Export"] * spend_breakdown_percentage["Netherlands"],
    "Vietnam to USA": annual_spend["US Air Import"] * spend_breakdown_percentage["Vietnam"]
}

# Provided tonnage for each route
tonnage = {
    "China to USA": 28571,
    "Taiwan to USA": 142851,
    "USA to Netherlands": 1200,
    "Vietnam to USA": 3571
}

# Calculate cost per ton for each route
cost_per_ton = {route: annual_spend_by_route[route] / tonnage[route] for route in tonnage}

# Convert cost per ton to cost per kilogram
cost_per_kg = {route: cost / 1000 for route, cost in cost_per_ton.items()}

# Output the results
print("Annual Spend by Route:", annual_spend_by_route)
print("Cost Per Ton:", cost_per_ton)
print("Cost Per Kilogram:", cost_per_kg)


# In[25]:


# Define a function to print the results in a formatted table
def print_formatted_results(annual_spend_by_route, cost_per_ton, cost_per_kg):
    print("{:<20} | {:<15} | {:<15} | {:<15}".format("Route", "Annual Spend ($)", "Cost/Ton ($)", "Cost/Kg ($)"))
    print("-" * 70)
    for route in annual_spend_by_route.keys():
        print("{:<20} | {:<15,.2f} | {:<15,.2f} | {:<15,.2f}".format(
            route, 
            annual_spend_by_route[route], 
            cost_per_ton[route], 
            cost_per_kg[route]))

print_formatted_results(annual_spend_by_route, cost_per_ton, cost_per_kg)




