from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.api.deps import require_current_user
from backend.app.models.user import User
from backend.app.schemas.user import UserResponse, UserUpdate
from backend.app.core.security import get_password_hash

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/me", response_model=UserResponse, summary="Get current authenticated user profile")
def get_me(user: User = Depends(require_current_user)):
    return user

@router.patch("/me", response_model=UserResponse, summary="Update current user profile")
def update_me(
    update_data: UserUpdate,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db)
):
    if update_data.name is not None:
        user.name = update_data.name
    if update_data.email is not None:
        user.email = update_data.email.lower()
    if update_data.password is not None:
        user.password_hash = get_password_hash(update_data.password)

    db.commit()
    db.refresh(user)
    return user
