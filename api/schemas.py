from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# Allowed values, kept in one place so the contract and the docs agree.
PropertyType = Literal['Flat', 'Independent Builder Floor', 'Independent House']
Furnishing = Literal['unfurnished', 'semi-furnished', 'furnished']
AgePossession = Literal[
    'New Property', 'Relatively New', 'Moderately Old',
    'Old Property', 'Under Construction'
]
Flag = Literal[0, 1]      # a yes/no amenity toggle


class PredictRequest(BaseModel):
    """What the caller sends: only things a person actually knows.

    Everything else the model needs (landmark distances, floor defaults,
    weak features) is derived server-side in api/inference.py.
    """

    # --- the essentials ---
    property_type: Annotated[PropertyType, Field(description="Type of property")]
    sector: Annotated[str, Field(description="Gurgaon sector", examples=["sector 49"])]
    area: Annotated[float, Field(gt=100, le=27000, description="Super built-up area (sqft)")]
    bedRoom: Annotated[int, Field(ge=1, le=10, description="Number of bedrooms")]
    bathroom: Annotated[int, Field(ge=1, le=10, description="Number of bathrooms")]

    # --- optional details (sensible defaults if omitted) ---
    furnishing: Annotated[Furnishing, Field(default='semi-furnished')]
    age_possession_category: Annotated[AgePossession, Field(default='New Property')]
    covered_parking: Annotated[int, Field(default=1, ge=0, le=10, description="Covered parking spaces")]
    total_floor: Annotated[
        Optional[int],
        Field(default=None, ge=1, le=90,
              description="Total floors in the building. Omit for independent houses.")
    ]

    # --- amenities ---
    has_ac: Annotated[Flag, Field(default=0, description="1 if air conditioning")]
    has_power_backup: Annotated[Flag, Field(default=0, description="1 if power backup")]
    has_pool: Annotated[Flag, Field(default=0, description="1 if swimming pool")]
    is_corner: Annotated[Flag, Field(default=0, description="1 if corner property")]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "property_type": "Flat",
                "sector": "sector 49",
                "area": 1500,
                "bedRoom": 3,
                "bathroom": 2,
                "furnishing": "semi-furnished",
                "age_possession_category": "New Property",
                "covered_parking": 1,
                "total_floor": 15,
                "has_ac": 1,
                "has_power_backup": 1,
                "has_pool": 0,
                "is_corner": 0,
            }
        }
    )


class PredictResponse(BaseModel):
    """What we send back. All prices in Indian Crores."""

    predicted_price_cr: Annotated[float, Field(description="Point estimate (Cr)")]
    lower_bound_cr: Annotated[float, Field(description="Lower bound of the 90% range (Cr)")]
    upper_bound_cr: Annotated[float, Field(description="Upper bound of the 90% range (Cr)")]
    model_name: Annotated[str, Field(description="Deployed model")]
    mape_percent: Annotated[float, Field(description="Model test MAPE (%)")]

    # 'model_' is reserved by pydantic; this lets us keep the field name.
    model_config = ConfigDict(protected_namespaces=())
