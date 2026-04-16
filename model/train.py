import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load data from train.csv
df = pd.read_csv('train.csv')
X = df[['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'GarageCars', 'KitchenQual', 'HouseStyle', 'LotArea', 'Neighborhood', 'ExterQual', 'BedroomAbvGr']]
y = df['SalePrice']

# 2. Define columns
num_cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'GarageCars', 'LotArea', 'BedroomAbvGr']
cat_cols = ['KitchenQual', 'HouseStyle', 'Neighborhood', 'ExterQual']

# 3. Create Preprocessing
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# 4. Create and Train Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))
])

model_pipeline.fit(X, y)

# 5. Save the new model
joblib.dump(model_pipeline, "house_price_model.pkl")
print("✅ Success! house_price_model.pkl has been retrained and saved.")