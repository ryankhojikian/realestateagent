import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

# ALL 12 REQUIRED FEATURES
data = {
    "OverallQual": [7],
    "GrLivArea": [1800],
    "TotalBsmtSF": [1000],
    "FullBath": [2],
    "YearBuilt": [2005],
    "GarageCars": [2],
    "KitchenQual": ["Gd"],       # Added
    "HouseStyle": ["1Story"],    # Added
    "LotArea": [8500],           # Added
    "Neighborhood": ["CollgCr"], # Added
    "ExterQual": ["Gd"],         # Added
    "BedroomAbvGr": [3]          # Added
}

test_df = pd.DataFrame(data)

try:
    prediction = model.predict(test_df)
    print(f"✅ SUCCESS! Predicted Price: ${prediction[0]:,.2f}")
except Exception as e:
    print(f"❌ Still missing something: {e}")


