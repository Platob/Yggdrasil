from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TypeAlias, Union

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.iam import ComplexValue, Group as GroupV1, User as UserV1
from databricks.sdk.service.iamv2 import Group as GroupV2, User as UserV2

from .service import IAMGroups, IAMUsers
from ..client import DatabricksResource

__all__ = [
    "IAMGroup",
    "IAMGroupLike",
    "IAMUser",
    "IAMUserLike",
]


IAMUserLike: TypeAlias = Union[
    str,
    "IAMUser",
    UserV1,
    UserV2,
    ComplexValue,
    Mapping[str, Any],
]

IAMGroupLike: TypeAlias = Union[
    str,
    "IAMGroup",
    GroupV1,
    GroupV2,
    ComplexValue,
    Mapping[str, Any],
]


def _coalesce(*values: Any) -> Any:
    """Return the first non-empty value."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


@dataclass
class IAMUser(DatabricksResource):
    """
    Lightweight user wrapper around Databricks IAM user payloads.

    Supports parsing from SDK v1/v2 models, SCIM complex values, mappings,
    and simple strings.
    """

    service: IAMUsers = field(
        default_factory=IAMUsers.current,
        repr=False,
        compare=False,
    )
    id: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    emails: Optional[list[str]] = None
    external_id: Optional[str] = None
    active: bool = True
    client_type: ClientType = ClientType.ACCOUNT

    @property
    def email(self) -> Optional[str]:
        return self.emails[0] if self.emails else None

    def __str__(self) -> str:
        return self.name or self.username or self.id or "unknown-user"

    def __repr__(self):
        n = self.name or self.username or self.id or "unknown-user"
        return f"{self.__class__.__name__}<{n!r}>"

    @classmethod
    def databricks_runtime(cls) -> "IAMUser":
        """Return the synthetic Databricks Runtime principal."""
        return cls(
            id="databricks-runtime",
            name="Databricks Runtime",
            username="databricks-runtime",
            client_type=ClientType.ACCOUNT,
            active=True,
        )

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        service: IAMUsers | None = None,
        client_type: Optional[ClientType] = None,
    ) -> "IAMUser":
        """
        Parse an arbitrary user-like object into ``IAMUser``.
        """
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, (UserV1, UserV2, ComplexValue)):
            return cls(
                service=service or IAMUsers.current(),
                client_type=client_type,
            ).set_details(obj)

        if isinstance(obj, Mapping):
            return cls.parse_mapping(
                obj,
                service=service,
                client_type=client_type,
            )

        if isinstance(obj, str):
            return cls.parse_str(
                obj,
                service=service,
                client_type=client_type,
            )

        raise ValueError(f"Unsupported object type for parsing IAMUser: {type(obj)}")

    @classmethod
    def parse_str(
        cls,
        value: str,
        *,
        service: IAMUsers | None = None,
        client_type: Optional[ClientType] = None,
    ) -> "IAMUser":
        """
        Parse a string into ``IAMUser``.

        Email-like strings are stored in ``emails`` and ``username``.
        Non-email strings are treated as usernames.
        """
        if not value:
            raise ValueError("Value cannot be empty for parsing IAMUser from string")

        is_email = "@" in value

        if client_type is None:
            client_type = service.client.default_client_type

        return cls(
            service=service or IAMUsers.current(),
            username=value,
            emails=[value] if is_email else None,
            client_type=client_type,
        )

    @classmethod
    def parse_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        service: IAMUsers | None = None,
        client_type: Optional[ClientType] = None,
        **kwargs: Any,
    ) -> "IAMUser":
        """
        Parse a mapping into ``IAMUser``.

        Accepts mixed v1/v2/SCIM-style field names.
        """
        if data:
            kwargs.update(data)

        user_id = _coalesce(kwargs.get("id"), kwargs.get("internal_id"), kwargs.get("value"))
        name = _coalesce(kwargs.get("display_name"), kwargs.get("name"), kwargs.get("display"))
        username = _coalesce(kwargs.get("username"), kwargs.get("user_name"), kwargs.get("email"))
        external_id = kwargs.get("external_id")
        emails = kwargs.get("emails")

        if isinstance(emails, str):
            emails = [emails]

        if not emails and username and "@" in username:
            emails = [username]

        if client_type is None:
            client_type = service.client.default_client_type

        return cls(
            service=service or IAMUsers.current(),
            id=user_id,
            name=name,
            username=username,
            emails=emails,
            external_id=external_id,
            client_type=client_type,
        )

    def set_details(self, details: "UserV1 | UserV2 | ComplexValue | IAMUser") -> "IAMUser":
        """
        Populate this instance from another user-like object.
        """
        if isinstance(details, IAMUser):
            self.id = details.id
            self.name = details.name
            self.username = details.username
            self.emails = details.emails
            self.external_id = details.external_id
            self.active = details.active
            self.client_type = details.client_type
            return self

        if isinstance(details, UserV1):
            self.id = details.id
            self.name = details.display_name
            self.username = _coalesce(getattr(details, "user_name", None), self.username)
            self.external_id = details.external_id
            self.emails = [item.value for item in details.emails] if details.emails else None
            self.active = True if details.active is None else details.active
            return self

        if isinstance(details, UserV2):
            self.id = details.internal_id
            self.name = None
            self.username = details.username
            self.external_id = details.external_id
            self.emails = [details.username] if details.username and "@" in details.username else None
            self.active = True
            return self

        if isinstance(details, ComplexValue):
            self.id = details.value
            self.name = details.display
            self.username = details.display
            self.external_id = None
            self.emails = [details.display] if details.display and "@" in details.display else None
            self.active = True
            return self

        raise ValueError(f"Unsupported user details type: {type(details)}")

    def sync(self) -> "IAMUser":
        """
        Refresh user details from Databricks.

        If only ``username`` is known, resolve the user id first via the service.
        """
        if not self.id:
            if not self.username:
                raise ValueError("User must have an ID or username to be synced")

            found = next(self.service.list(user_name=self.username), None)
            if found is None:
                raise ResourceDoesNotExist(
                    f"User with username '{self.username}' not found for syncing"
                )
            return self.set_details(found)

        if self.client_type == ClientType.WORKSPACE:
            details = self.client.workspace_client().users.get(id=self.id)
        elif self.client_type == ClientType.ACCOUNT:
            details = self.client.account_client().users.get(id=self.id)
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

        return self.set_details(details)

    @property
    def complex_value(self) -> ComplexValue:
        """
        Return this user as a SCIM ``ComplexValue``.
        """
        if not self.id:
            self.sync()

        return ComplexValue(
            value=self.id,
            display=self.name or self.username,
            ref=f"Users/{self.id}" if self.id else None,
        )


@dataclass
class IAMGroup(DatabricksResource):
    """
    Lightweight group wrapper around Databricks IAM group payloads.
    """

    service: IAMGroups = field(
        default_factory=IAMGroups.current,
        repr=False,
        compare=False,
    )
    id: Optional[str] = None
    name: Optional[str] = None
    account_id: Optional[str] = None
    external_id: Optional[str] = None
    client_type: Optional[ClientType] = None
    entitlements: Optional[list[str]] = None
    members: Optional[list[IAMUser]] = None

    def __post_init__(self) -> None:
        if self.service is None:
            object.__setattr__(self, "service", IAMGroups.current())
        super().__post_init__()

    def __str__(self) -> str:
        return self.name or self.id or "unknown-group"

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        service: IAMGroups | None = None,
        client_type: Optional[ClientType] = None,
    ) -> "IAMGroup":
        """
        Parse an arbitrary group-like object into ``IAMGroup``.
        """
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, (GroupV1, GroupV2, ComplexValue)):
            return cls(
                service=service or IAMGroups.current(),
                client_type=client_type,
            ).set_details(obj)

        if isinstance(obj, Mapping):
            return cls.parse_mapping(
                obj,
                service=service,
                client_type=client_type,
            )

        raise ValueError(f"Unsupported object type for parsing IAMGroup: {type(obj)}")

    @classmethod
    def parse_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        service: IAMGroups | None = None,
        client_type: Optional[ClientType] = None,
        **kwargs: Any,
    ) -> "IAMGroup":
        """
        Parse a mapping into ``IAMGroup``.

        Accepts mixed v1/v2/SCIM-style field names.
        """
        if data:
            kwargs.update(data)

        service = service or IAMGroups.current()

        group_id = _coalesce(kwargs.get("id"), kwargs.get("internal_id"), kwargs.get("value"))
        name = _coalesce(kwargs.get("display_name"), kwargs.get("name"), kwargs.get("display"))
        external_id = kwargs.get("external_id")
        account_id = kwargs.get("account_id")
        raw_members = kwargs.get("members") or kwargs.get("users") or []

        members = [
            IAMUser.parse(member, service=service.users)
            for member in raw_members
            if member
        ] or None

        return cls(
            service=service,
            client_type=client_type,
            id=group_id,
            name=name,
            external_id=external_id,
            account_id=account_id,
            members=members,
        )

    def set_details(self, details: "GroupV1 | GroupV2 | ComplexValue | IAMGroup") -> "IAMGroup":
        """
        Populate this instance from another group-like object.
        """
        if isinstance(details, IAMGroup):
            self.id = details.id
            self.name = details.name
            self.external_id = details.external_id
            self.account_id = details.account_id
            self.client_type = details.client_type
            self.entitlements = details.entitlements
            self.members = details.members
            return self

        if isinstance(details, GroupV1):
            self.id = details.id
            self.name = details.display_name
            self.external_id = details.external_id
            self.account_id = self.client.config.account_id
            self.entitlements = [item.value for item in details.entitlements or ()]

            if details.meta and details.meta.resource_type:
                self.client_type = (
                    ClientType.ACCOUNT
                    if details.meta.resource_type == "Group"
                    else ClientType.WORKSPACE
                )

            self.members = [
                IAMUser.parse(member, service=self.service.users)
                for member in details.members or []
            ] or None
            return self

        if isinstance(details, GroupV2):
            self.id = details.internal_id
            self.name = details.group_name
            self.external_id = details.external_id
            self.account_id = details.account_id
            self.members = None
            self.entitlements = None
            return self

        if isinstance(details, ComplexValue):
            self.id = details.value
            self.name = details.display
            self.external_id = None
            self.account_id = None
            self.members = None
            self.entitlements = None
            return self

        raise ValueError(f"Unsupported group details type: {type(details)}")

    def add_member(
        self,
        user: IAMUserLike,
        *,
        commit: bool = True,
    ) -> "IAMGroup":
        """
        Add a user to the group if not already present.

        Membership is deduplicated by id, then by name, then by username.
        """
        parsed_user = IAMUser.parse(user, service=self.service.users)

        if self.members is None:
            self.members = []

        if not parsed_user.id and not parsed_user.name and not parsed_user.username:
            parsed_user = parsed_user.sync()
        elif not parsed_user.id and parsed_user.username:
            parsed_user = parsed_user.sync()

        if self._has_member(parsed_user):
            return self

        self.members.append(parsed_user)

        if commit:
            self.sync()

        return self

    def _has_member(self, user: IAMUser) -> bool:
        """
        Return whether the provided user is already a member of the group.
        """
        if not self.members:
            return False

        if user.id and any(member.id == user.id for member in self.members):
            return True

        if user.name and any(member.name == user.name for member in self.members):
            return True

        if user.username and any(member.username == user.username for member in self.members):
            return True

        return False

    def sync(self) -> "IAMGroup":
        """
        Persist group updates to Databricks.
        """
        if not self.id:
            raise ValueError("Group must have an ID to be committed")

        payload_members = [member.complex_value for member in self.members] if self.members else None

        if self.client_type == ClientType.WORKSPACE:
            details = self.client.workspace_client().groups.update(
                id=self.id,
                display_name=self.name,
                external_id=self.external_id or None,
                members=payload_members,
            )
        elif self.client_type == ClientType.ACCOUNT:
            details = self.client.account_client().groups.update(
                id=self.id,
                display_name=self.name,
                external_id=self.external_id or None,
                members=payload_members,
            )
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

        return self.set_details(details)