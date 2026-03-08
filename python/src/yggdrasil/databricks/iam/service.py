import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Literal, Sequence

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.iam import ResourceMeta

from ..client import DatabricksService

if TYPE_CHECKING:
    from .resource import IAMGroup, IAMGroupLike, IAMUser, IAMUserLike

__all__ = [
    "IAM",
    "IAMGroups",
    "IAMUsers",
]


@dataclass(frozen=True)
class IAM(DatabricksService):

    @property
    def groups(self):
        return self.client.lazy_property(
            self,
            cache_attr="_groups",
            factory=lambda: IAMGroups(client=self.client),
            use_cache=True
        )

    @property
    def users(self):
        return self.client.lazy_property(
            self,
            cache_attr="_users",
            factory=lambda: IAMUsers(client=self.client),
            use_cache=True
        )


@dataclass(frozen=True)
class IAMGroups(IAM):

    def create(
        self,
        name: str,
        *,
        resource_type: Literal["account", "workspace"] = "account",
        client_type: Optional[ClientType] = None,
        members: Optional[Sequence["IAMUserLike"]] = None,
        group: Optional["IAMGroupLike"] = None,
        **kwargs
    ) -> "IAMGroup":
        from .resource import IAMGroup

        if client_type is None:
            client_type = self.client.default_client_type

        if not members:
            members = [
                self.client.iam.users.current_user
            ]

        if group is None:
            group = IAMGroup.parse_mapping(
                kwargs,
                service=self,
                client_type=client_type,
                name=name,
                members=members
            )
        else:
            group = IAMGroup.parse(
                group,
                service=self,
                client_type=client_type
            )

        if resource_type == "account":
            meta = ResourceMeta(resource_type="Group")
        elif resource_type == "workspace":
            meta = ResourceMeta(resource_type="WorkspaceGroup")
        else:
            raise ValueError(f"Invalid resource_type: {resource_type}")

        members = [_.complex_value for _ in group.members] if group.members else None

        if client_type == ClientType.ACCOUNT:
            details = (
                self.client.account_client()
                .groups.create(
                    display_name=group.name,
                    external_id=group.external_id or None,
                    id=group.id or None,
                    members=members,
                    meta=meta,
                )
            )
        else:
            details = (
                self.client.workspace_client()
                .groups.create(
                    display_name=group.name,
                    external_id=group.external_id or None,
                    id=group.id or None,
                    members=members,
                    meta=meta,
                )
            )

        return IAMGroup.parse(
            details,
            service=self,
            client_type=client_type
        )

    def list(
        self,
        *,
        name: Optional[str] = None,
        client_type: Optional[ClientType] = None,
        limit: Optional[int] = None,
        raise_error: bool = True
    ) -> Iterator["IAMGroup"]:
        from .resource import IAMGroup

        if client_type is None:
            client_type = self.client.config.client_type

        if client_type == ClientType.ACCOUNT:
            client = self.client.account_client()
        else:
            client = self.client.workspace_client()

        if name:
            filter_by = f'displayName eq "{name}"'
        else:
            filter_by = None

        cnt, limit = 0, limit or float("inf")

        try:
            for details in client.groups.list(filter=filter_by):
                group = IAMGroup.parse(details, service=self, client_type=client_type)

                yield group
                cnt += 1

                if cnt >= limit:
                    break
        except DatabricksError:
            if raise_error:
                raise


@dataclass(frozen=True)
class IAMUsers(IAM):

    def list(
        self,
        *,
        name: Optional[str] = None,
        user_name: Optional[str] = None,
        client_type: Optional[ClientType] = None,
        limit: Optional[int] = None
    ) -> Iterator["IAMUser"]:
        from .resource import IAMUser

        if client_type is None:
            client_type = self.client.config.client_type

        if client_type == ClientType.ACCOUNT:
            client = self.client.account_client()
        else:
            client = self.client.workspace_client()

        filter_by = []

        if name:
            filter_by.append(f'displayName eq "{name}"')

        if user_name:
            filter_by.append(f'userName eq "{user_name}"')

        filter_by = " and ".join(filter_by) if filter_by else None

        cnt, limit = 0, limit or float("inf")

        for details in client.users.list(
            filter=filter_by
        ):
            user = IAMUser.parse(details, service=self, client_type=client_type)

            yield user
            cnt += 1

            if cnt >= limit:
                break

    @property
    def current_user(self) -> "IAMUser":
        from .resource import IAMUser

        def factory():
            try:
                details = self.client.workspace_client().current_user.me()

                return IAMUser.parse(
                    details,
                    service=self,
                    client_type=self.client.default_client_type
                )
            except DatabricksError:
                if self.client.auth_type == "external-browser":
                    self.reset_local_cache()
                    raise
                elif self.client.auth_type == "runtime":
                    return IAMUser.databricks_runtime()
                else:
                    raise

        return self.client.lazy_property(
            self,
            cache_attr="_current_user",
            factory=factory,
            use_cache=True
        )

    def local_cache_token_path(self):
        oauth_dir = self.client.local_config_folder / "oauth"
        if not oauth_dir.is_dir():
            return None

        # "first" = lexicographically first (stable)
        files = sorted(p for p in oauth_dir.iterdir() if p.is_file())
        return str(files[0]) if files else None

    def reset_local_cache(self):
        """Remove cached browser OAuth tokens.

        Returns:
            None.
        """
        local_cache = self.local_cache_token_path()

        if local_cache:
            os.remove(local_cache)