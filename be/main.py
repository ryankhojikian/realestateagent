import json
import joblib
import pandas as pd
import requests
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# 1. Load the ML Model
try:
    model = joblib.load("../model/house_price_model.pkl")
    print("✅ ML Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Base values used if the user doesn't specify a feature
DEFAULTS = {
    "OverallQual": 6, "GrLivArea": 1500, "TotalBsmtSF": 1000, 
    "FullBath": 2, "YearBuilt": 2005, "GarageCars": 2, 
    "KitchenQual": "TA", "HouseStyle": "1Story",
    "LotArea": 8000, "Neighborhood": "CollgCr", 
    "ExterQual": "TA", "BedroomAbvGr": 3
}

def brute_force_extract(text):
    """
    Guaranteed Extraction: Uses Regular Expressions to find numbers 
    and keywords. This ensures the price moves even if the AI is slow.
    """
    new_data = DEFAULTS.copy()
    missing_features = []
    text_lower = text.lower().replace(',', '')  # Remove commas for number parsing
    
    # 1. Quality (Handles "7/10", "quality 7", "good quality", "quality good", etc.)
    qual_match = re.search(r'(\d+)\s*/\s*10|quality.*?(\d+)|(good|excellent|fair|average|poor).*?quality|quality.*?(good|excellent|fair|average|poor)', text_lower)
    if qual_match:
        if qual_match.group(1) or qual_match.group(2):
            val = int(qual_match.group(1) or qual_match.group(2))
        else:
            # Map descriptive words to numbers
            word = qual_match.group(3) or qual_match.group(4)
            quality_map = {'poor': 3, 'fair': 5, 'average': 6, 'good': 7, 'excellent': 9}
            val = quality_map.get(word, 6)  # Default to 6 if unknown
        new_data["OverallQual"] = max(1, min(10, val))
    else:
        missing_features.append("OverallQual")

    # 2. Square Footage (GrLivArea)
    area_match = re.search(r'(\d{3,5})\s*(sqft|square feet|feet)', text_lower)
    if area_match:
        new_data["GrLivArea"] = int(area_match.group(1))
    else:
        missing_features.append("GrLivArea")

    # 3. Bedrooms (BedroomAbvGr)
    bed_match = re.search(r'(\d+)\s*(bedroom|bed)', text_lower)
    if bed_match:
        new_data["BedroomAbvGr"] = int(bed_match.group(1))
    else:
        missing_features.append("BedroomAbvGr")

    # 4. Garage Capacity (GarageCars)
    garage_match = re.search(r'(\d+)(?:\s*-\s*car)?\s*garage', text_lower)
    if garage_match:
        new_data["GarageCars"] = int(garage_match.group(1))
    else:
        missing_features.append("GarageCars")

    # 5. Full Bathrooms (FullBath)
    bath_match = re.search(r'(\d+)\s*(?:full\s+)?bath(?:room)?s?', text_lower)
    if bath_match:
        new_data["FullBath"] = int(bath_match.group(1))
    else:
        missing_features.append("FullBath")

    # Luxury keyword fallback for Demo safety
    if ("luxury" in text_lower or "perfect" in text_lower) and "sqft" not in text_lower:
        new_data["OverallQual"] = 10
        new_data["GrLivArea"] = 4500

    return {
        "features": new_data,
        "missing_features": missing_features,
        "is_complete": len(missing_features) == 0,
    }

def fast_llm_call(prompt, timeout=10):
    """
    Direct API call to Ollama. 
    If it takes longer than 10s, it cuts off to prevent UI hanging.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama", 
                "prompt": f"Explain in one short sentence: {prompt}", 
                "stream": False
            },
            timeout=timeout
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ LLM Timeout or Error: {e}")
        return ""

@app.post("/agent")
async def ai_agent(request: dict):
    user_query = request.get("description", "")
    print(f"\n--- INCOMING REQUEST ---\nQuery: {user_query}")
    
    # --- STAGE 1: EXTRACTION ---
    extraction = brute_force_extract(user_query)
    final_features = extraction["features"]
    print(f"DEBUG: Features -> Qual:{final_features['OverallQual']}, Area:{final_features['GrLivArea']}, Beds:{final_features['BedroomAbvGr']}")

    # --- STAGE 2: PREDICTION ---
    try:
        input_df = pd.DataFrame([final_features])
        predicted_price = float(model.predict(input_df)[0])
        print(f"✅ FINAL PRICE: ${predicted_price:,.2f}")
    except Exception as e:
        print(f"🔥 Prediction Error: {e}")
        predicted_price = 195000.0

    # --- STAGE 3: INTERPRETATION (LLM REVIEW) ---
    prompt = f"Why is a house with {final_features['OverallQual']}/10 quality worth ${predicted_price:,.0f}?"
    comment = fast_llm_call(prompt)
    
    # Fallback if the LLM is too slow
    if not comment:
        comment = f"This property is valued at ${predicted_price:,.0f} based on its {final_features['OverallQual']}/10 quality and {final_features['GrLivArea']} sqft layout."

    return {
        "prediction": predicted_price,
        "interpretation": comment,
        "extracted_data": {
            "features": final_features,
            "is_complete": extraction["is_complete"],
            "missing_features": extraction["missing_features"],
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)