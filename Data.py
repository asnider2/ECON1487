import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
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
columns_for_regression = ['men', 'material_category', 'gender', 'pricesamountmin', 'size_numeric']

# Filtering the DataFrame
regression_data = shoe_data[columns_for_regression]

# Drop rows with NaN values in the target variable
regression_data = regression_data.dropna(subset=['pricesamountmin'])

# Mapping the gender column
regression_data['gender'] = regression_data['gender'].map({'men': 1, 'women': 0})

# Handling categorical variables (material_category)
regression_data = pd.get_dummies(regression_data, columns=['material_category'], dummy_na=True)

# Splitting the data into features (X) and target variable (y)
X = regression_data.drop('pricesamountmin', axis=1)
y = regression_data['pricesamountmin']

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Splitting the data into training and testing sets with imputed data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)

# Predict prices on the test set
y_pred = model.predict(X_test)

# Compare predicted prices for men and women
predicted_prices = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Gender': X_test[:, columns_for_regression.index('gender')],  # Extract gender column from X_test
    'Size': X_test[:, columns_for_regression.index('size_numeric')]  # Extract size_numeric column from X_test
})

# Visualize the predicted prices for men and women in a scatter plot
plt.figure(figsize=(18, 6))

# Scatter plot for gender
plt.subplot(1, 3, 1)
sns.scatterplot(x='Actual', y='Predicted', hue='Gender', data=predicted_prices)
plt.title('Scatter Plot for Gender')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# Line graph for sizes
plt.subplot(1, 3, 2)
sns.lineplot(x='Size', y='Predicted', data=predicted_prices)
plt.title('Line Graph for Size')
plt.xlabel('Size')
plt.ylabel('Predicted Prices')

# Line graph for gender
plt.subplot(1, 3, 3)
sns.lineplot(x='Gender', y='Predicted', data=predicted_prices)
plt.title('Line Graph for Gender')
plt.xlabel('Gender')
plt.ylabel('Predicted Prices')

plt.tight_layout()
plt.show()
