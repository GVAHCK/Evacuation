from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.app.core.security import decode_token
from backend.app.database.session import get_db
from backend.app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

def authenticate_token_string(token: str, db: Session) -> Optional[User]:
    """Decodes token and retrieves user. Returns None if invalid or user does not exist."""
    if not token:
        return None
    payload = decode_token(token, is_refresh=False)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()

def get_current_user(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[User]:
    """Retrieves authenticated user from JWT bearer token. Returns None or default user if unauthenticated."""
    if not token:
        return None

    payload = decode_token(token, is_refresh=False)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN", "message": "Could not validate authentication credentials"}
        )

    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN_PAYLOAD", "message": "Token missing subject"}
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "USER_NOT_FOUND", "message": "User no longer exists"}
        )

    return user

def require_current_user(user: Optional[User] = Depends(get_current_user)) -> User:
    """Enforces active authenticated session."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTHENTICATION_REQUIRED", "message": "Authentication token required"}
        )
    return user

def require_researcher_or_admin(user: User = Depends(require_current_user)) -> User:
    """Enforces researcher or admin role permissions."""
    if user.role not in ("RESEARCHER", "ADMIN"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Researcher or Admin privileges required"}
        )
    return user
