from pydantic import BaseModel, Field
from typing import List, Optional

class HouseFeatures(BaseModel):
    OverallQual: Optional[int] = Field(None, ge=1, le=10)
    GrLivArea: Optional[int] = None
    TotalBsmtSF: Optional[int] = None
    FullBath: Optional[int] = None
    YearBuilt: Optional[int] = None
    GarageCars: Optional[int] = None
    KitchenQual: Optional[str] = None
    HouseStyle: Optional[str] = None
    LotArea: Optional[int] = None
    Neighborhood: Optional[str] = None
    ExterQual: Optional[str] = None
    BedroomAbvGr: Optional[int] = None

class PredictionResponse(BaseModel):
    prediction: float
    interpretation: str
    extracted_data: dict