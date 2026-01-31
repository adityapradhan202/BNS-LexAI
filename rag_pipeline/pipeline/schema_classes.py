from pydantic import BaseModel
from typing import Literal

class QueryClassify(BaseModel):
    query_type:Literal["related", "not-related"]

class ResponseClassify(BaseModel):
    relevancy:Literal["relevant", "irrelevant"]