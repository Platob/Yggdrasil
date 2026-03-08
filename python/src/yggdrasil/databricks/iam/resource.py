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
    client_type: ClientType = ClientType.ACCOUNT
    members: list["IAMUser"] = field(default_factory=list)

    def __str__(self):
        return self.name

    def __post_init__(self):
        if self.service is None:
            object.__setattr__(self, "service", IAMGroups.current())

        if self.client_type is None:
            object.__setattr__(self, "client_type", ClientType.ACCOUNT)

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
            group = IAMGroup(
                service=service,
                client_type=client_type
            )
            group.set_details(obj)
            return group

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
            self.commit()

        return self

    def commit(self) -> "IAMGroup":
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

        self.set_details(details)

        return self

    def set_details(self, details: GroupV1 | GroupV2):
        if isinstance(details, GroupV1):
            object.__setattr__(self, "id", details.id)
            object.__setattr__(self, "name", details.display_name)
            object.__setattr__(self, "external_id", details.external_id)
            object.__setattr__(self, "account_id", self.client.config.account_id)

            members = [
                IAMUser.parse(
                    member,
                    service=self.service.users,
                    client_type=self.client_type
                )
                for member in details.members or []
            ]
            object.__setattr__(self, "members", members)

        elif isinstance(details, GroupV2):
            object.__setattr__(self, "id", details.internal_id)
            object.__setattr__(self, "name", details.group_name)
            object.__setattr__(self, "external_id", details.external_id)
            object.__setattr__(self, "account_id", details.account_id)

            if self.id and details.internal_id != self.id:
                object.__setattr__(self, "members", [])

        elif isinstance(details, ComplexValue):
            object.__setattr__(self, "id", details.value)
            object.__setattr__(self, "name", details.display)
            object.__setattr__(self, "account_id", self.client.config.account_id)

            if self.id and details.value != self.id:
                object.__setattr__(self, "external_id", None)
                object.__setattr__(self, "members", [])

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
    email: Optional[str] = None
    external_id: Optional[str] = None
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
            email=""
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
            email = value
            name = value.split("@")[0]
        else:
            email = None
            name = value

        return cls(
            service=service,
            id=None,
            name=name,
            email=email,
            client_type=client_type or ClientType.ACCOUNT
        )

    @classmethod
    def parse_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        service: IAMUsers | None = None,
        client_type: Optional[ClientType] = None,
    ):
        group_id = data.get("id") or data.get("internal_id") or data.get("value")
        name = data.get("display_name") or data.get("name") or data.get("display")
        external_id = data.get("external_id")

        user = cls(
            service=service,
            id=group_id,
            name=name,
            external_id=external_id,
            client_type=client_type or ClientType.ACCOUNT
        )

        return user

    def set_details(self, details: UserV1 | UserV2 | ComplexValue):
        if isinstance(details, UserV1):
            object.__setattr__(self, "id", details.id)
            object.__setattr__(self, "name", details.display_name)
            object.__setattr__(self, "external_id", details.external_id)

            email = details.emails[0].value if details.emails else None
            if not self.email:
                object.__setattr__(self, "email", email)
            elif email and email != self.email:
                object.__setattr__(self, "email", email)

        elif isinstance(details, UserV2):
            object.__setattr__(self, "id", details.internal_id)
            object.__setattr__(self, "name", details.name)
            object.__setattr__(self, "external_id", details.external_id)
            object.__setattr__(self, "email", details.username)

        elif isinstance(details, ComplexValue):
            object.__setattr__(self, "id", details.value)
            object.__setattr__(self, "name", details.display)
            object.__setattr__(self, "external_id", None)

            if self.id and details.value != self.id:
                object.__setattr__(self, "email", None)

        else:
            raise ValueError(f"Unsupported user details type: {type(details)}")

    def sync(self):
        if not self.id:
            if self.username:
                found = next(self.service.list(user_name=self.username), None)

                if found is None:
                    raise ResourceDoesNotExist(
                        f"User with username '{self.username}' not found for syncing"
                    )
                return found
            else:
                raise ValueError("User must have an ID to be synced")

        if self.client_type == ClientType.WORKSPACE:
            details = self.client.workspace_client().users.get(id=self.id)
        elif self.client_type == ClientType.ACCOUNT:
            details = self.client.account_client().users.get(id=self.id)
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

        self.set_details(details)
        return self

    @property
    def username(self):
        return self.email

    @property
    def complex_value(self) -> ComplexValue:
        if not self.id:
            self.sync()
        return ComplexValue(
            value=self.id,
            display=self.name,
            ref=f"Users/{self.id}" if self.id else None
        )
