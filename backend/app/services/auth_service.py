from datetime import timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.core.config import settings
from backend.app.core.security import verify_password, get_password_hash, create_access_token, create_refresh_token, decode_token
from backend.app.models.user import User
from backend.app.schemas.auth import LoginRequest, RegisterRequest, Token

class AuthService:
    """Manages user registration, credential validation, and JWT token issuance."""

    @staticmethod
    def register_user(req: RegisterRequest, db: Session) -> User:
        existing = db.query(User).filter(User.email == req.email.lower()).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "EMAIL_ALREADY_EXISTS", "message": "Email is already registered"}
            )

        role = req.role.upper() if req.role and req.role.upper() in ("USER", "RESEARCHER", "ADMIN") else "USER"
        user = User(
            name=req.name,
            email=req.email.lower(),
            password_hash=get_password_hash(req.password),
            role=role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def authenticate_user(req: LoginRequest, db: Session) -> Tuple[User, Token]:
        user = db.query(User).filter(User.email == req.email.lower()).first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "INVALID_CREDENTIALS", "message": "Invalid email or password"}
            )

        access_token = create_access_token({"sub": user.id, "role": user.role})
        refresh_token = create_refresh_token({"sub": user.id, "role": user.role})

        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        return user, token

    @staticmethod
    def refresh_access_token(refresh_token: str, db: Session) -> Token:
        payload = decode_token(refresh_token, is_refresh=True)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "INVALID_REFRESH_TOKEN", "message": "Refresh token is invalid or expired"}
            )

        user_id = payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "USER_NOT_FOUND", "message": "User no longer exists"}
            )

        new_access = create_access_token({"sub": user.id, "role": user.role})
        new_refresh = create_refresh_token({"sub": user.id, "role": user.role})

        return Token(
            access_token=new_access,
            refresh_token=new_refresh,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    @staticmethod
    def seed_default_users(db: Session):
        """Seeds default user, researcher, and admin accounts if they do not exist (development/testing only)."""
        if settings.ENVIRONMENT.lower() == "production":
            return
        seeds = [
            ("Evacuation User", "user@evacuation.ai", "user123", "USER"),
            ("Dr. RL Researcher", "researcher@evacuation.ai", "researcher123", "RESEARCHER"),
            ("System Admin", "admin@evacuation.ai", "admin123", "ADMIN")
        ]
        for name, email, pwd, role in seeds:
            if not db.query(User).filter(User.email == email).first():
                u = User(
                    name=name,
                    email=email,
                    password_hash=get_password_hash(pwd),
                    role=role
                )
                db.add(u)
        db.commit()

auth_service = AuthService()
