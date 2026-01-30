from pydantic import BaseModel
from typing import Literal

class BetterSearchType(BaseModel):
    type:Literal[1,2]