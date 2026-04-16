from pydantic import BaseModel, Field
from typing import Annotated, Literal


class PredictRequest(BaseModel):
    """
    Raw inputs from the caller.
    Derived features (area_per_bedroom, plot_area_missing) are
    computed inside the API — callers never send them.
    """

    property_type : Annotated[Literal['Flat', 'Independent House', 'Independent Builder Floor'], Field(..., description="Type of property")]

    society : Annotated[str, Field(..., description="Society / apartment complex name")]

    sector : Annotated[str, Field(..., description="Sector Name", examples=['sector 50'])]

    total_area_sqft : Annotated[float, Field(..., gt=100, le=15000, description="Total Area in SQFT")]

    bedrooms : Annotated[int, Field(..., ge=1, le=10, description="Number of Bedrooms")]

    bathrooms : Annotated[int, Field(..., ge=1, le=10, description="Number of Bathrooms")]

    balcony : Annotated[str, Field(default='0', description="Number of Balconies (0, 1, 2, 3...)")]

    servant_room : Annotated[Literal[0, 1], Field(default=0, description="1 if servant room present, else 0")]

    pooja_room : Annotated[Literal[0, 1], Field(default=0, description="1 if pooja room present, else 0")]

    facing : Annotated[str, Field(default='east', description="Direction the property faces", examples=['east', 'north', 'north-east'])]

    furnishing_type : Annotated[Literal['Furnished', 'Semi-Furnished', 'Unfurnished'], Field(default='Unfurnished', description="Furnishing status")]

    age_possession : Annotated[Literal['New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction', 'Undefined'], Field(default='Relatively New', description="Age / possession category")]

    luxury_category : Annotated[Literal['Budget', 'Semi-Luxury', 'Luxury'], Field(default='Budget', description="Luxury tier")]


class PredictResponse(BaseModel):
    """
    Prediction output returned to the caller.
    All prices are in Indian Crores (₹ Cr).
    """

    predicted_price_cr : Annotated[float, Field(..., description="Point estimate of property price in Crores")]

    lower_bound_cr : Annotated[float, Field(..., description="Lower bound of price range based on model MAPE")]

    upper_bound_cr : Annotated[float, Field(..., description="Upper bound of price range based on model MAPE")]

    model_name : Annotated[str, Field(..., description="Name of the ML model used for prediction")]

    mape_percent : Annotated[float, Field(..., description="Model test MAPE — indicates prediction uncertainty")]


class HealthResponse(BaseModel):

    status : Annotated[str, Field(..., description="API status")]

    model_loaded : Annotated[bool, Field(..., description="Whether model artifact is loaded successfully")]

    model_name : Annotated[str, Field(..., description="Name of the loaded model")]

    mape_percent : Annotated[float, Field(..., description="Model test MAPE")]