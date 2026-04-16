from __future__ import annotations

import dataclasses as dc
import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Union

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.catalog import (
    EffectivePermissionsList,
    EffectivePrivilegeAssignment,
    GetPermissionsResponse,
    PermissionsChange,
    Privilege,
    PrivilegeAssignment,
    SecurableType,
    UpdatePermissionsResponse,
)

from yggdrasil.databricks.client import DatabricksResource, DatabricksService

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from .table import Table
    from .schema import Schema
    from .column import Column
    from .catalog import Catalog

__all__ = [
    "Grant",
    "Grants",
]

LOGGER = logging.getLogger(__name__)


def _safe_enum(enum_cls: type[Any], value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value

    for candidate in (
        lambda: enum_cls(value),
        lambda: enum_cls(value.value),
        lambda: enum_cls[value.name],
        lambda: enum_cls[str(value)],
    ):
        try:
            return candidate()
        except Exception:
            pass

    if isinstance(value, str):
        lowered_value = value.casefold()

        for candidate in enum_cls:
            enum_value = str(candidate.value).casefold()
            enum_name = str(candidate.name).casefold()

            if lowered_value == enum_value or lowered_value == enum_name:
                return candidate
            if lowered_value in enum_value or lowered_value in enum_name:
                return candidate

    raise ValueError(f"Cannot coerce {value!r} into {enum_cls.__name__}")


def _check_securable_type(value: SecurableType | str) -> SecurableType:
    if value is None:
        raise ValueError("securable_type must be non-None")
    return _safe_enum(SecurableType, value)


def _check_privilege(value: Privilege | str) -> Privilege:
    if value is None:
        raise ValueError("privilege must be non-None")
    return _safe_enum(Privilege, value)


def _check_privileges(values: Sequence[Privilege | str] | None) -> list[Privilege]:
    if not values:
        return []
    return [_check_privilege(value) for value in values]


def _check_principal(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("principal must be non-empty")
    return value


@dc.dataclass
class Grant(DatabricksResource):
    service: "Grants"
    securable_type: SecurableType | str
    full_name: str
    principal: str
    privileges: tuple[str, ...] = ()
    effective: bool = False

    @property
    def client(self) -> "DatabricksClient":
        return self.service.client

    @property
    def securable_type_checked(self) -> SecurableType:
        return _check_securable_type(self.securable_type)

    def refresh(self, *, effective: bool | None = None) -> "Grant":
        return self.service.get(
            securable_type=self.securable_type_checked,
            full_name=self.full_name,
            principal=self.principal,
            effective=self.effective if effective is None else effective,
            raise_error=True,
        )

    def grant(
        self,
        privileges: Sequence[Privilege | str],
    ) -> "Grant":
        self.service.update(
            securable_type=self.securable_type_checked,
            full_name=self.full_name,
            changes=[
                PermissionsChange(
                    principal=self.principal,
                    add=_check_privileges(privileges),
                )
            ],
        )
        return self.refresh(effective=False)

    def revoke(
        self,
        privileges: Sequence[Privilege | str],
    ) -> "Grant":
        self.service.update(
            securable_type=self.securable_type_checked,
            full_name=self.full_name,
            changes=[
                PermissionsChange(
                    principal=self.principal,
                    remove=_check_privileges(privileges),
                )
            ],
        )
        return self.refresh(effective=False)

    def replace(
        self,
        privileges: Sequence[Privilege | str],
    ) -> "Grant":
        target = {_check_privilege(value) for value in privileges}
        current = {_check_privilege(value) for value in self.privileges}

        to_add = sorted(target - current, key=lambda value: value.value)
        to_remove = sorted(current - target, key=lambda value: value.value)

        if not to_add and not to_remove:
            return self.refresh(effective=False)

        self.service.update(
            securable_type=self.securable_type_checked,
            full_name=self.full_name,
            changes=[
                PermissionsChange(
                    principal=self.principal,
                    add=to_add or None,
                    remove=to_remove or None,
                )
            ],
        )
        return self.refresh(effective=False)

    def delete(self) -> None:
        if not self.privileges:
            return

        self.service.update(
            securable_type=self.securable_type_checked,
            full_name=self.full_name,
            changes=[
                PermissionsChange(
                    principal=self.principal,
                    remove=_check_privileges(self.privileges),
                )
            ],
        )


@dc.dataclass(frozen=True)
class Grants(DatabricksService):
    catalog_name: str | None = None
    schema_name: str | None = None

    @classmethod
    def service_name(cls) -> str:
        return "grants"

    @property
    def _api(self):
        return self.client.workspace_client().grants

    @staticmethod
    def _iter_assignments(
        response: GetPermissionsResponse | EffectivePermissionsList,
    ) -> Iterator[PrivilegeAssignment | EffectivePrivilegeAssignment]:
        yield from response.privilege_assignments or []

    @staticmethod
    def _assignment_principal(
        assignment: PrivilegeAssignment | EffectivePrivilegeAssignment,
    ) -> str | None:
        return getattr(assignment, "principal", None)

    @staticmethod
    def _assignment_privileges(
        assignment: PrivilegeAssignment | EffectivePrivilegeAssignment,
    ) -> tuple[str, ...]:
        values: list[str] = []

        for item in getattr(assignment, "privileges", None) or []:
            value = getattr(item, "value", None)
            values.append(value or str(item))

        return tuple(values)

    def _read_page(
        self,
        *,
        securable_type: SecurableType | str,
        full_name: str,
        principal: str | None,
        max_results: int | None,
        page_token: str | None,
        effective: bool,
    ) -> GetPermissionsResponse | EffectivePermissionsList:
        checked_type = _check_securable_type(securable_type)

        kwargs = {
            "securable_type": checked_type.value,
            "full_name": full_name,
            "principal": principal,
            "max_results": max_results,
            "page_token": page_token,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if effective:
            return self._api.get_effective(**kwargs)
        return self._api.get(**kwargs)

    def list(
        self,
        obj: Union[
            str,
            "Catalog", "Schema", "Table", "Column",
            None,
        ] = None,
        *,
        securable_type: SecurableType | str | None = None,
        full_name: str | None = None,
        principal: str | None = None,
        effective: bool = False,
        max_results: int | None = 0,
    ) -> Iterator[Grant]:
        if obj is not None:
            try:
                cls_name = obj.__class__.__name__.lower()
            except Exception as e:
                raise ValueError(f"Cannot determine securable type for {obj!r}") from e

            if hasattr(obj, "full_name") and callable(getattr(obj, "full_name", None)):
                full_name = obj.full_name(safe=False)

            securable_type = _check_securable_type(cls_name)

        checked_type = _check_securable_type(securable_type)
        page_token: str | None = None

        while True:
            response = self._read_page(
                securable_type=checked_type,
                full_name=full_name,
                principal=principal,
                max_results=max_results,
                page_token=page_token,
                effective=effective,
            )

            for assignment in self._iter_assignments(response):
                assignment_principal = self._assignment_principal(assignment)
                if not assignment_principal:
                    continue

                yield Grant(
                    service=self,
                    securable_type=checked_type,
                    full_name=full_name,
                    principal=assignment_principal,
                    privileges=self._assignment_privileges(assignment),
                    effective=effective,
                )

            next_page_token = getattr(response, "next_page_token", None)
            if not next_page_token:
                break
            page_token = next_page_token

    def get(
        self,
        *,
        securable_type: SecurableType | str,
        full_name: str,
        principal: str,
        effective: bool = False,
        raise_error: bool = True,
    ) -> Grant | None:
        principal = _check_principal(principal)

        for grant in self.list(
            securable_type=securable_type,
            full_name=full_name,
            principal=principal,
            effective=effective,
            max_results=0,
        ):
            if grant.principal == principal:
                return grant

        if raise_error:
            raise ResourceDoesNotExist(
                f"Cannot find grant for principal {principal!r} on "
                f"{_check_securable_type(securable_type).value} {full_name!r}"
            )
        return None

    def create(
        self,
        *,
        securable_type: SecurableType | str,
        full_name: str,
        principal: str,
        privileges: Sequence[Privilege | str],
    ) -> Grant:
        principal = _check_principal(principal)

        self.update(
            securable_type=securable_type,
            full_name=full_name,
            changes=[
                PermissionsChange(
                    principal=principal,
                    add=_check_privileges(privileges),
                )
            ],
        )

        return self.get(
            securable_type=securable_type,
            full_name=full_name,
            principal=principal,
            effective=False,
            raise_error=True,
        )

    def update(
        self,
        *,
        securable_type: SecurableType | str,
        full_name: str,
        changes: Sequence[PermissionsChange] | None = None,
    ) -> UpdatePermissionsResponse:
        checked_type = _check_securable_type(securable_type)

        LOGGER.debug(
            "Updating grants on %s %r with %d change(s)",
            checked_type.value,
            full_name,
            len(changes or ()),
        )

        return self._api.update(
            securable_type=checked_type.value,
            full_name=full_name,
            changes=list(changes or ()) or None,
        )