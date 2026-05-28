from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_user_service
from ..schemas.user import UserCard, UserListResponse
from ..services.user import UserService

router = APIRouter(tags=["user"])


@router.get("/me", response_model=UserCard)
async def get_me(
    service: UserService = Depends(get_user_service),
) -> UserCard:
    return service.get_self()


@router.get("", response_model=UserListResponse)
async def list_users(
    service: UserService = Depends(get_user_service),
) -> UserListResponse:
    return service.list()


@router.post("/register", response_model=UserCard)
async def register_user(
    card: UserCard,
    service: UserService = Depends(get_user_service),
) -> UserCard:
    return service.register(card)
