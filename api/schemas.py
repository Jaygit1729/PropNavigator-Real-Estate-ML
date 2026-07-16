from pydantic import BaseModel, Field
from typing import Annotated, Literal


class PredictRequest(BaseModel):
    """
    Raw inputs from the caller.
    Derived features (area_per_bedroom, plot_area_missing) are
    computed inside the API — callers never send them.
    """
    
    property_type: Annotated[Literal['Flat', 'Independent Builder Floor', 'Independent House'], Field(description="Type of property")]
    sector: Annotated[str, Field(description="Gurgaon sector", examples=["sector 49"])]
    area: Annotated[float, Field(gt=100, le=27000, description="Area in sqft")]
    bedRoom: Annotated[int, Field(ge=1, le=10, description="Number of bedrooms")]
    bathroom: Annotated[int, Field(ge=1, le=10, description="Number of bathrooms")]