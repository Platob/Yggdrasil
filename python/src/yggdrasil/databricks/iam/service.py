import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, Union

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import DatabricksError, PermissionDenied
from databricks.sdk.service.iam import ResourceMeta

from ..client import DatabricksService

if TYPE_CHECKING:
    from .resource import IAMGroup, IAMGroupLike, IAMUser, IAMUserLike

__all__ = [
    "IAM",
    "IAMGroups",
    "IAMUsers",
]

logger = logging.getLogger(__name__)


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
        client_type: Optional[ClientType] = None,
        members: Optional[Sequence["IAMUserLike"]] = None,
        group: Optional["IAMGroupLike"] = None,
        **kwargs
    ) -> "IAMGroup":
        from .resource import IAMGroup

        if client_type is None:
            client_type = self.client.default_client_type

        if not members:
            members = [self.client.iam.users.current_user]

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

        if client_type == ClientType.ACCOUNT:
            meta = ResourceMeta(resource_type="Group")
        else: # Fallback to workspace group resource type for compatibility with older SDK versions
            meta = ResourceMeta(resource_type="WorkspaceGroup")

        members = [_.complex_value for _ in group.members] if group.members else None

        logger.debug(
            "Creating IAM group name=%s client_type=%s",
            group.name,
            client_type,
        )

        try:
            if client_type == ClientType.ACCOUNT:
                details = (
                    self.client.account_client()
                    .groups.create(
                        display_name=group.name,
                        external_id=group.external_id,
                        id=group.id,
                        members=members,
                        meta=meta,
                    )
                )
            else:
                details = (
                    self.client.workspace_client()
                    .groups.create(
                        display_name=group.name,
                        external_id=group.external_id,
                        id=group.id,
                        members=members,
                        meta=meta,
                    )
                )
        except PermissionDenied as e:
            raise PermissionDenied(
                f"Failed to create IAM group '{name}' with client_type '{client_type}'. "
                "This may be due to insufficient permissions. "
                "Please ensure that your credentials have the necessary permissions to create groups in the specified context.",
                error_code=e.error_code,
                details=e.details,
            ) from e

        result = IAMGroup.parse(
            details,
            service=self,
            client_type=client_type
        )

        logger.info(
            "Created IAM group name=%s id=%s client_type=%s",
            result.name,
            result.id,
            client_type,
        )
        return result

    def delete(
        self,
        obj: Union["IAMGroup", str],
        *,
        group_id: Optional[str] = None,
        client_type: Optional[ClientType] = None
    ):
        if isinstance(obj, IAMGroup):
            return self.delete_group(group=obj, client_type=client_type)
        elif group_id:
            return self.delete_group(group=group_id, client_type=client_type)
        else:
            raise ValueError("Either obj or group_id must be provided")

    def delete_group(
        self,
        group: Union["IAMGroup", str],
        *,
        client_type: Optional[ClientType] = None
    ):
        if isinstance(group, IAMGroup):
            group_id = group.id

            if client_type is None:
                client_type = group.client_type
        else:
            group_id = str(group)

        logger.debug(
            "Deleting IAM group id=%s client_type=%s",
            group_id,
            client_type,
        )

        if client_type is None:
            client_type = self.client.default_client_type

        if client_type == ClientType.ACCOUNT:
            self.client.account_client().groups.delete(group_id)
        else:
            self.client.workspace_client().groups.delete(group_id)

        logger.info(
            "Deleted IAM group id=%s client_type=%s",
            group_id,
            client_type,
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

        filter_by = f'displayName eq "{name}"' if name else None
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

    def create(
        self,
        name: str,
        *,
        active: bool = True,
        client_type: Optional[ClientType] = None,
        user: Optional["IAMUserLike"] = None,
        **kwargs
    ):
        from .resource import IAMUser

        if user is None:
            user = IAMUser.parse_mapping(
                kwargs,
                service=self,
                client_type=client_type,
                name=name,
                active=active
            )
        else:
            user = IAMUser.parse(
                user,
                service=self,
                client_type=client_type
            )

        client_type = client_type or self.client.default_client_type

        logger.debug(
            "Creating IAM user name=%s active=%s client_type=%s",
            user.name,
            user.active,
            client_type,
        )

        try:
            if client_type == ClientType.ACCOUNT:
                details = (
                    self.client.account_client()
                    .users.create(
                        display_name=user.name,
                        external_id=user.external_id,
                        id=user.id,
                        active=user.active,
                    )
                )
            else:
                details = (
                    self.client.workspace_client()
                    .users.create(
                        display_name=user.name,
                        external_id=user.external_id,
                        id=user.id,
                        active=user.active,
                    )
                )
        except Exception:
            raise

        result = IAMUser.parse(
            details,
            service=self,
            client_type=client_type
        )

        logger.info(
            "Created IAM user name=%s id=%s client_type=%s",
            result.name,
            result.id,
            client_type,
        )
        return result

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

        for details in client.users.list(filter=filter_by):
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
                result = IAMUser.parse(
                    details,
                    service=self,
                    client_type=self.client.default_client_type
                )
                logger.debug("Got current IAM user name=%s id=%s", result.name, result.id)
                return result
            except DatabricksError as e:
                if self.client.auth_type == "external-browser":
                    self.reset_local_cache()
                    raise
                else:
                    logger.exception(e)
                    return IAMUser.databricks_runtime()

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

        files = sorted(p for p in oauth_dir.iterdir() if p.is_file())
        return str(files[0]) if files else None

    def reset_local_cache(self):
        """Remove cached browser OAuth tokens.

        Returns:
            None.
        """
        logger.debug("Resetting local OAuth cache")

        local_cache = self.local_cache_token_path()

        if local_cache:
            os.remove(local_cache)
            logger.info("Reset local OAuth cache path=%s", local_cache)
        else:
            logger.info("No local OAuth cache found to reset")