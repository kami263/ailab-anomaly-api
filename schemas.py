from pydantic import BaseModel, Field
from typing import List

class AnomalyRequest(BaseModel):
    values: List[float] = Field(..., min_items=1, description="時系列データ")
