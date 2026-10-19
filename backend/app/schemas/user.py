from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class UserBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=3, max_length=150)
    role: str = "USER"

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[str] = Field(None, min_length=3, max_length=150)
    password: Optional[str] = Field(None, min_length=6)

class UserResponse(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

