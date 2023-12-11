import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from pprint import pprint
from sklearn.metrics import *
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor 



df = pd.read_csv("googleplaystore.csv")




df.columns = df.columns.str.strip()

columns_list = df.columns.tolist()
print("Columns in the spreadsheet:")
print(columns_list)

column_name = 'Category'

# Using unique():
unique_categories = df[column_name].unique()
print("Unique categories using unique():")
print(unique_categories)

df['Category_Code'] = df['Category'].map({'ART_AND_DESIGN': 1, 'AUTO_AND_VEHICLES': 2, 'BEAUTY': 3, 'BOOKS_AND_REFERENCE': 4, 'BUSINESS': 5, 'COMICS': 6, 'COMMUNICATION': 7, 'DATING': 8, 'EDUCATION': 9, 'ENTERTAINMENT': 10, 'EVENTS': 11, 'FAMILY': 12, 'FINANCE': 13, 'FOOD_AND_DRINK': 14, 'GAME': 15, 'HEALTH_AND_FITNESS': 16, 'HOUSE_AND_HOME': 17, 'LIBRARIES_AND_DEMO': 18, 'LIFESTYLE': 19, 'MAPS_AND_NAVIGATION': 20, 'MEDICAL': 21, 'NEWS_AND_MAGAZINES': 22, 'PARENTING': 23, 'PERSONALIZATION': 24, 'PHOTOGRAPHY': 25, 'PRODUCTIVITY': 26, 'SHOPPING': 27, 'SOCIAL': 28, 'SPORT': 29, 'TOOLS': 30, 'TRAVEL_AND_LOCAL': 31, 'VIDEO_PLAYERS': 32, 'WEATHER': 33})

df['Content Rating_Code'] = df['Content Rating'].map({'Adults only 18+': 1, 'Everyone': 2, 'Everyone 10+': 3,'Mature 17+': 4, 'Teen': 5})

df['Genres_Code'] = df['Genres'].map({'Action': 1, 'Adventure': 2, 'Arcade': 3, 'Art & Design': 4, 'Auto & Vehicles': 5, 'Beauty': 6, 'Board': 7, 'Books & Reference': 8, 'Business': 9, 'Card': 10, 'Casino': 11, 'Casual': 12, 'Comics': 13, 'Communication': 14, 'Dating': 15, 'Education': 16, 'Educational': 17, 'Entertainment': 18, 'Events': 19, 'Finance': 20, 'Food & Drink': 21, 'Health & Fitness': 22, 'House & Home': 23, 'Libraries & Demo': 24, 'Lifestyle': 25, 'Maps & Navigation': 26, 'Medical': 27, 'Music': 28, 'Music & Audio': 29,'News & Magazines': 30, 'Parenting': 31, 'Personalization': 32, 'Photography': 33, 'Productivity': 34, 'Puzzle': 35, 'Racing': 36, 'Role Playing': 37, 'Shopping': 38, 'Simulation': 39, 'Social': 40, 'Sports': 41, 'Strategy': 42, 'Tools': 43, 'Travel & Local': 44, 'Trivia': 45, 'Video Players & Editors': 46, 'Weather': 47, 'Word': 48})

df['Type_Code'] = df['Type'].map({'Free': 1, 'Paid': 2})

print(df.loc[:, ["Category_Code", "Content Rating_Code", "Genres_Code"]].head(20))

np.random.seed(10) #Same results


columns_to_drop = ['Current Ver', 'Android Ver'] 

df.drop(columns=columns_to_drop, inplace=True)



# Replace 'Varies with device' with NaN across all columns
df.replace('Varies with device', pd.NA, inplace=True)

def value_to_float(x):
    if pd.isna(x):  # Check if x is NaN
        return x  # Return NaN as is
    if type(x) == float or type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0

# Apply the modified function to the 'Size' column
df['Size'] = df['Size'].apply(value_to_float)
df = df.dropna(subset=['Size'])
df.dropna(inplace=True)

# Remove '+' from the 'Numbers' column
df['Installs'] = df['Installs'].str.replace('+', '')

# Remove commas from the 'Numbers' column
df['Installs'] = df['Installs'].str.replace(',', '')

# Convert the 'Numbers' column to integers
df['Installs'] = df['Installs'].astype(int)

df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)

df.to_csv("clean.csv") # exporting to a new CSV

# Independent variables/features
list_cols_X = ["Price", "Content Rating_Code", "Genres_Code"]

# Dependent variable/target
col_y = "Rating"

# Keep only interested variables
df = df[list_cols_X + [col_y]]

# Shuffle the data set randomly
df = df.sample(frac=1).reset_index(drop=True)

# Create training data set
df_train = df.loc[:700, :].copy().reset_index(drop=True)

# Create evaluation data set
df_eval = df.loc[701:, :].copy().reset_index(drop=True)
print("Number records in training data: " + str(len(df_train)))
print("Number records in evaluation data: " + str(len(df_eval)))



# Create a untrained neural network model
model = MLPRegressor(verbose=True)

# Train the model
model.fit(df_train[list_cols_X], df_train[col_y])


# Print model detail
pprint(vars(model))

# Produce predictions by testing the model on evaluation data set
list_predicted = model.predict(df_eval[list_cols_X])
# Place true and predicted values of the targe in a new dataframe
df_target = pd.DataFrame()
df_target["target_true"] = df_eval[col_y].copy()
df_target["target_pred"] = list_predicted.copy()
print("Compare true and predicted values:\n" + str(df_target.head(10)))
print("Describe true and predicted values:\n" + str(df_target.describe()))



score_MAPE = mean_absolute_percentage_error(df_eval[col_y], list_predicted)
score_MAE = mean_absolute_error(df_eval[col_y], list_predicted)
score_MSE = mean_squared_error(df_eval[col_y], list_predicted)
score_ExpVar = explained_variance_score(df_eval[col_y], list_predicted)
score_R2 = r2_score(df_eval[col_y], list_predicted)
print("Mean absolute percentage error: " + str(score_MAPE))
print("Mean absolute error: " + str(score_MAE))
print("Mean squared error: " + str(score_MSE))
print("Explained variance: " + str(score_ExpVar))
print("R2: " + str(score_R2))



num_pacs = 3
df_temp = pd.DataFrame(PCA(n_components=num_pacs).fit(df_train[list_cols_X]).transform(df_train[list_cols_X])).copy()
df_temp[col_y] = df_train[col_y].copy()
df_train = df_temp.copy()
print(df_train.head(5))
df_temp = pd.DataFrame(PCA(n_components=num_pacs).fit(df_eval[list_cols_X]).transform(df_eval[list_cols_X])).copy()
df_temp[col_y] = df_eval[col_y].copy()
df_eval = df_temp.copy()
print(df_eval.head(5))
list_cols_X = [col for col in range(0, num_pacs)]

model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=0.001)


