from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class APIRequest(BaseModel):
    start_row :int
    start_col :int
    fire_location:List[str]
    exits:List[str]
    stage:str

class APIResponse(BaseModel):
    path: List[str]
    length: float
    download_url:str
    