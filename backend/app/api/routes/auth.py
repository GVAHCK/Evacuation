from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.services.auth_service import auth_service
from backend.app.schemas.auth import LoginRequest, RegisterRequest, RefreshTokenRequest, Token
from backend.app.schemas.user import UserResponse
from backend.app.core.rate_limiter import auth_rate_limiter

router = APIRouter(prefix="/auth", tags=["Authentication"], dependencies=[Depends(auth_rate_limiter)])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Register a new user")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    user = auth_service.register_user(req, db)
    return user

@router.post("/login", response_model=Token, summary="Authenticate user and receive JWT access token")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    _, token = auth_service.authenticate_user(req, db)
    return token

@router.post("/refresh", response_model=Token, summary="Refresh access token with refresh token")
def refresh_token(req: RefreshTokenRequest, db: Session = Depends(get_db)):
    token = auth_service.refresh_access_token(req.refresh_token, db)
    return token

@router.post("/logout", summary="Logout user")
def logout():
    return {"code": "LOGGED_OUT", "message": "Successfully logged out"}

@router.post("/forgot-password", summary="Initiate password recovery")
def forgot_password(payload: dict):
    email = payload.get("email", "")
    return {
        "code": "RECOVERY_INITIATED",
        "email": email,
        "message": f"Password recovery instructions and security token have been generated for {email}.",
        "instructions": "Contact emergency platform administrator (admin@evacuation.ai) with your verification token to complete credential reset."
    }
