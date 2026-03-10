from dataclasses import field, dataclass
from typing import Any, Optional, Mapping, Union

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.iam import Group as GroupV1, ComplexValue, User as UserV1
from databricks.sdk.service.iamv2 import Group as GroupV2, User as UserV2

from .service import IAMGroups, IAMUsers
from ..client import DatabricksResource

__all__ = [
    "IAMGroup",
    "IAMGroupLike",
    "IAMUser",
    "IAMUserLike"
]


IAMUserLike = Union[str, "IAMUser", UserV1, UserV2, ComplexValue, Mapping[str, Any]]
IAMGroupLike = Union[str, "IAMGroup", GroupV1, GroupV2, ComplexValue, Mapping[str, Any]]


@dataclass
class IAMGroup(DatabricksResource):
    service: IAMGroups = field(
        default_factory=IAMGroups.current,
        repr=False,
        compare=False
    )
    id: Optional[str] = None
    name: Optional[str] = None
    account_id: Optional[str] = None
    external_id: Optional[str] = None
    client_type: Optional[ClientType] = None
    entitlements: Optional[list[str]] = None
    members: Optional[list["IAMUser"]] = None

    def __str__(self):
        return self.name

    def __post_init__(self):
        if self.service is None:
            object.__setattr__(self, "service", IAMGroups.current())

        super().__post_init__()

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        service: IAMGroups | None = None,
        client_type: Optional[ClientType] = None,
    ):
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, (GroupV1, GroupV2, ComplexValue)):
            return IAMGroup(
                service=service,
                client_type=client_type
            ).set_details(obj)

        elif isinstance(obj, Mapping):
            return cls.parse_mapping(obj, service=service, client_type=client_type)

        else:
            raise ValueError(f"Unsupported object type for parsing IAMGroup: {type(obj)}")

    @classmethod
    def parse_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        service: IAMGroups | None = None,
        client_type: Optional[ClientType] = None,
        **kwargs
    ):
        if data:
            kwargs.update(data)

        group_id = kwargs.get("id") or kwargs.get("internal_id") or kwargs.get("value")
        display_name = kwargs.get("display_name") or kwargs.get("name") or kwargs.get("display")
        external_id = kwargs.get("external_id")
        account_id = kwargs.get("account_id")
        members = kwargs.get("members") or kwargs.get("users") or []

        if members:
            members = [
                IAMUser.parse(member, service=service.users)
                for member in members
                if member
            ]

        group = cls(
            service=service,
            client_type=client_type,
            id=group_id,
            name=display_name,
            external_id=external_id,
            account_id=account_id,
            members=members
        )

        return group

    def add_member(
        self,
        user: IAMUserLike,
        *,
        commit: bool = True
    ) -> "IAMGroup":
        user = IAMUser.parse(user, service=self.service.users)

        if user.id:
            if any(member.id == user.id for member in self.members):
                return self
        elif user.name:
            if any(member.name == user.name for member in self.members):
                return self
        elif user.username:
            if any(member.username == user.username for member in self.members):
                return self
        else:
            user = user.sync()

            if any(member.id == user.id for member in self.members):
                return self

        self.members.append(user)

        if commit:
            self.sync()

        return self

    def sync(self) -> "IAMGroup":
        assert self.id, "Group must have an ID to be committed"

        if self.client_type == ClientType.WORKSPACE:
            details = self.client.workspace_client().groups.update(
                id=self.id,
                display_name=self.name,
                external_id=self.external_id or None,
                members=[
                    member.complex_value
                    for member in self.members
                ] if self.members else None
            )

        elif self.client_type == ClientType.ACCOUNT:
            details = self.client.account_client().groups.update(
                id=self.id,
                display_name=self.name,
                external_id=self.external_id or None,
                members=[member.complex_value for member in self.members] if self.members else None
            )

        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

        return self.set_details(details)

    def set_details(self, details: Union[GroupV1, GroupV2, "IAMGroup"]) -> "IAMGroup":
        if isinstance(details, GroupV1):
            self.id = details.id
            self.name = details.display_name
            self.external_id = details.external_id
            self.account_id = self.client.config.account_id
            self.entitlements = [_.value for _ in details.entitlements or ()]

            if details.meta:
                if details.meta.resource_type:
                    self.client_type = ClientType.ACCOUNT if details.meta.resource_type == "Group" else ClientType.WORKSPACE

            self.members = [
                IAMUser.parse(
                    member,
                    service=self.service.users,
                )
                for member in details.members or []
            ]

        elif isinstance(details, GroupV2):
            self.id = details.internal_id
            self.name = details.group_name
            self.external_id = details.external_id
            self.account_id = details.account_id
            self.members = None
            self.entitlements = None

        elif isinstance(details, ComplexValue):
            self.id = details.value
            self.name = details.display
            self.external_id = None
            self.account_id = None
            self.members = None
            self.entitlements = None

        elif isinstance(details, IAMGroup):
            self.id = details.id
            self.name = details.name
            self.external_id = details.external_id
            self.account_id = details.account_id
            self.client_type = details.client_type
            self.entitlements = details.entitlements
            self.members = details.members

        else:
            raise ValueError(f"Unsupported group details type: {type(details)}")

        return self


@dataclass
class IAMUser(DatabricksResource):
    service: IAMUsers = field(
        default_factory=IAMUsers.current,
        repr=False,
        compare=False
    )
    id: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    emails: Optional[list[str]] = None
    external_id: Optional[str] = None
    active: bool = True
    client_type: ClientType = ClientType.ACCOUNT

    def __post_init__(self):
        if self.service is None:
            object.__setattr__(self, "service", IAMUsers.current())

        super().__post_init__()

    def __str__(self):
        return self.name

    @classmethod
    def databricks_runtime(cls):
        return cls(
            id="databricks-runtime",
            name="Databricks Runtime",
            username="databricks-runtime",
            client_type=ClientType.ACCOUNT,
            active=True
        )

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        service: IAMUsers | None = None,
        client_type: Optional[ClientType] = None
    ):
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, (UserV1, UserV2, ComplexValue)):
            user = cls(service=service, client_type=client_type or ClientType.ACCOUNT)
            user.set_details(obj)
            return user

        elif isinstance(obj, Mapping):
            return cls.parse_mapping(obj, service=service, client_type=client_type)

        elif isinstance(obj, str):
            return cls.parse_str(obj, service=service, client_type=client_type)

        else:
            raise ValueError(f"Unsupported object type for parsing IAMUser: {type(obj)}")

    @classmethod
    def parse_str(
        cls,
        value: str,
        *,
        service: IAMUsers | None = None,
        client_type: Optional[ClientType] = None
    ):
        if not value:
            raise ValueError("Value cannot be empty for parsing IAMUser from string")

        if "@" in value:
            emails = [value]
        else:
            emails = None

        return cls(
            service=service,
            id=None,
            name=None,
            emails=emails,
            client_type=client_type or ClientType.ACCOUNT
        )

    @classmethod
    def parse_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        service: IAMUsers | None = None,
        client_type: Optional[ClientType] = None,
        **kwargs
    ):
        if data:
            kwargs.update(data)

        group_id = kwargs.get("id") or kwargs.get("internal_id") or kwargs.get("value")
        name = kwargs.get("display_name") or kwargs.get("name") or kwargs.get("display")
        external_id = kwargs.get("external_id")

        user = cls(
            service=service,
            id=group_id,
            name=name,
            external_id=external_id,
            client_type=client_type or ClientType.ACCOUNT
        )

        return user

    def set_details(self, details: UserV1 | UserV2 | ComplexValue) -> "IAMUser":
        if isinstance(details, UserV1):
            self.id = details.id
            self.name = details.display_name
            self.external_id = details.external_id
            self.emails = [_.value for _ in details.emails] if details.emails else None
            self.active = True if details.active is None else details.active

        elif isinstance(details, UserV2):
            self.id = details.internal_id
            self.external_id = details.external_id
            self.name = None
            self.username = details.username
            self.active = True

            if "@" in details.username:
                self.emails = [details.username]
            else:
                self.emails = None

        elif isinstance(details, ComplexValue):
            self.id = details.value
            self.name = details.display
            self.username = details.display
            self.active = True

            if "@" in details.display:
                self.emails = [details.display]
            else:
                self.emails = None

        elif isinstance(details, IAMUser):
            self.id = details.id
            self.name = details.name
            self.username = details.username
            self.emails = details.emails
            self.external_id = details.external_id
            self.active = details.active

        else:
            raise ValueError(f"Unsupported user details type: {type(details)}")

        return self

    def sync(self) -> "IAMUser":
        if not self.id:
            if self.username:
                found = next(self.service.list(user_name=self.username), None)

                if found is None:
                    raise ResourceDoesNotExist(
                        f"User with username '{self.username}' not found for syncing"
                    )

                return self.set_details(details=found)
            else:
                raise ValueError("User must have an ID to be synced")

        if self.client_type == ClientType.WORKSPACE:
            details = self.client.workspace_client().users.get(id=self.id)
        elif self.client_type == ClientType.ACCOUNT:
            details = self.client.account_client().users.get(id=self.id)
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

        return self.set_details(details)

    @property
    def complex_value(self) -> ComplexValue:
        if not self.id:
            self.sync()
        return ComplexValue(
            value=self.id,
            display=self.name,
            ref=f"Users/{self.id}" if self.id else None
        )
