import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import re

# Importing data into dataframe
shoe_data = pd.DataFrame(pd.read_csv("/Users/angelasnider/Desktop/ECON1487/shoe_data.csv"))

# Constants
GENDER_VARIABLES = ['Men', 'Women', 'Unisex']

# Function to extract 'Material' from the 'features' column
def extract_material(features):
    try:
        feature_json = json.loads(features.replace("'", "\""))
        material = [feature['value'] for feature in feature_json if feature['key'] == "Material"]
        return ', '.join(sum(material, []))
    except:
        return None

# Apply the function to create a new 'material' column
shoe_data['material'] = shoe_data['features'].apply(extract_material)

# Cleaning Data

# Drops Duplicate Rows
shoe_data.drop_duplicates(inplace=True)

# Makes Sizes Separate Rows
shoe_data['sizes'] = shoe_data['sizes'].str.split(',')
shoe_data = shoe_data.explode('sizes')

# Make numeric
shoe_data['pricesamountmin'] = pd.to_numeric(shoe_data['pricesamountmin'], errors='coerce')
shoe_data['pricesamountmax'] = pd.to_numeric(shoe_data['pricesamountmax'], errors='coerce')

# Remove Unnecessary Columns
columns_to_drop = ['id', 'manufacturer', 'manufacturernumber']
shoe_data.drop(columns=columns_to_drop, inplace=True)

def extract_gender(categories):
    categories_lower = categories.lower()
    if 'women' in categories_lower:
        return 'women'
    elif 'men' in categories_lower:
        return 'men'
    elif 'unisex' in categories_lower:
        return 'unisex'
    else:
        return np.nan

# Apply the function to create a new 'gender' column
shoe_data['gender'] = shoe_data['categories'].apply(extract_gender)

# Drop rows where 'gender' is NaN
shoe_data = shoe_data.dropna(subset=['gender'])

def categorize_material(material):
    material = str(material).lower()

    if 'leather' in material:
        return 'Leather'
    elif 'synthetic' in material or 'polyurethane' in material or 'vinyl' in material or 'plastic' in material:
        return 'Synthetic'
    elif 'canvas' in material:
        return 'Canvas'
    elif 'suede' in material:
        return 'Suede'
    elif 'fabric' in material or 'nylon' in material or 'mesh' in material:
        return 'Fabric'
    elif 'shearling' in material or 'fur' in material:
        return 'Shearling/Fur'
    else:
        return 'Other'

# Apply function to create a new 'material_category' column
shoe_data['material_category'] = shoe_data['material'].apply(categorize_material)

# Print unique values in the 'material_category' column to verify
print(shoe_data['material_category'].unique())

# Print unique values in the 'gender' column to verify
print(shoe_data['gender'].unique())
print(shoe_data['material'].unique())

# Standardize color
valid_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black', 'white', 'pink']

def map_to_valid_color(colors):
    if pd.isna(colors):
        return 'other'  # or any default value you prefer for NaN

    # Split the string into a list of colors
    color_list = [color.strip().lower() for color in colors.split(',')]

    # Find the first valid color, default to 'other' if none is found
    for color in color_list:
        if color in valid_colors:
            return color

    return 'other'

shoe_data['colors'] = shoe_data['colors'].apply(map_to_valid_color)

# Drop unknown values
shoe_data = shoe_data[shoe_data['pricesamountmin'].notna()]

##calculations

# Create a new column 'size_numeric' by extracting the numeric part of 'sizes'
shoe_data['size_numeric'] = shoe_data['sizes'].apply(lambda x: re.search(r'\d+', str(x)).group() if re.search(r'\d+', str(x)) else np.nan)

# Convert the 'size_numeric' column to numeric values
shoe_data['size_numeric'] = pd.to_numeric(shoe_data['size_numeric'], errors='coerce')

# Specify the columns for regression
columns_for_regression = ['gender', 'material_category', 'pricesamountmin', 'size_numeric']

# Filtering the DataFrame
regression_data = shoe_data[columns_for_regression]

# Drop rows with NaN values in the target variable
regression_data = regression_data.dropna()

# One-hot encoding for 'gender'
regression_data = pd.get_dummies(regression_data, columns=['gender'], prefix='gender', drop_first=True)

# Handling categorical variables (material_category)
regression_data = pd.get_dummies(regression_data, columns=['material_category'], dummy_na=True)

# Splitting the data into features (X) and target variable (y)
X = regression_data.drop('pricesamountmin', axis=1)
y = regression_data['pricesamountmin']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)

# Get the coefficients
coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})

# Print the coefficients
print("Coefficients:")
print(coefficients)

# Predict prices on the test set
y_pred = model.predict(X_test)

# Print columns of X_test to check the correct column names
print("Columns of X_test:", X_test.columns)

# Compare predicted prices for men and women
predicted_prices = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Gender': X_test[X_test.filter(like='gender').columns[0]],  # Dynamic column selection
    'Size': X_test['size_numeric']
})

# Print columns of X_test to check the correct column names
print("Columns of X_test:", X_test.columns)

# Convert 'Gender' column to numeric for indexing
predicted_prices['Gender'] = pd.to_numeric(predicted_prices['Gender'], errors='coerce')

# Visualize the predicted prices for men and women in a scatter plot
plt.figure(figsize=(15, 6))

# Scatter plot for gender
plt.subplot(1, 3, 1)
sns.scatterplot(x='Actual', y='Predicted', hue='Gender', style='Gender', data=predicted_prices)
plt.title('Scatter Plot for Gender')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# Bar plot for coefficients
plt.subplot(1, 3, 2)
sns.lineplot(x='Coefficient', y='Variable', data=coefficients)  # Use lineplot instead of barplot
plt.title('Coefficients')
plt.xlabel('Coefficient Value')

# Group size categories into ranges
size_ranges = pd.cut(predicted_prices['Size'], bins=[0, 2, 5, 8, 11, 14], labels=['0-2', '3-5', '6-8', '9-11', '12-14'])

# Update the 'Size' column with the grouped size ranges
predicted_prices['Size'] = size_ranges

# Sort the DataFrame based on the grouped size ranges
predicted_prices.sort_values(by=['Size', 'Gender'], inplace=True)

# Aggregate the data by taking the mean of predicted prices for each group
predicted_prices_agg = predicted_prices.groupby(['Size', 'Gender']).mean().reset_index()

# Line graph for sizes
plt.figure(figsize=(15, 6))

# Line graph for sizes and gender
plt.subplot(1, 3, 1)
sns.lineplot(x='Size', y='Predicted', hue='Gender', data=predicted_prices_agg)
plt.title('Line Graph for Size and Gender')
plt.xlabel('Size Range')
plt.ylabel('Predicted Prices')

plt.tight_layout()
plt.show()
