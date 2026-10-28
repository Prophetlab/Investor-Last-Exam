"""Financial data lake package with lazily loaded public SDK 1 exports."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import (
        ApiError as ApiError,
    )
    from .client import (
        ApiKeyIssueRequest as ApiKeyIssueRequest,
    )
    from .client import (
        ApiKeyIssueResponse as ApiKeyIssueResponse,
    )
    from .client import (
        ApiKeySessionProvider as ApiKeySessionProvider,
    )
    from .client import (
        Capabilities as Capabilities,
    )
    from .client import (
        ContentResponse as ContentResponse,
    )
    from .client import (
        DataLakeClient as DataLakeClient,
    )
    from .client import (
        DataLakeClientEnvironment as DataLakeClientEnvironment,
    )
    from .client import (
        DiscoveryResponse as DiscoveryResponse,
    )
    from .client import (
        DocumentDetail as DocumentDetail,
    )
    from .client import (
        DownloadReceipt as DownloadReceipt,
    )
    from .client import (
        Hit as Hit,
    )
    from .client import (
        ImageContent as ImageContent,
    )
    from .client import (
        LookupRequest as LookupRequest,
    )
    from .client import (
        Period as Period,
    )
    from .client import (
        ReadRequest as ReadRequest,
    )
    from .client import (
        SearchRequest as SearchRequest,
    )
    from .client import (
        SessionInfo as SessionInfo,
    )
    from .client import (
        TableCatalog as TableCatalog,
    )
    from .client import (
        TableDownloadReceipt as TableDownloadReceipt,
    )
    from .client import (
        TableResult as TableResult,
    )
    from .client import (
        TaskTokenIssueRequest as TaskTokenIssueRequest,
    )
    from .client import (
        TaskTokenIssueResponse as TaskTokenIssueResponse,
    )
    from .client import (
        TrustedTaskTokenIssuer as TrustedTaskTokenIssuer,
    )
    from .client import (
        TrustedTaskTokenProvider as TrustedTaskTokenProvider,
    )

__version__ = "1.1.0"

__all__ = [
    "ApiError",
    "ApiKeyIssueRequest",
    "ApiKeyIssueResponse",
    "ApiKeySessionProvider",
    "Capabilities",
    "ContentResponse",
    "DataLakeClient",
    "DataLakeClientEnvironment",
    "DiscoveryResponse",
    "DocumentDetail",
    "DownloadReceipt",
    "Hit",
    "ImageContent",
    "LookupRequest",
    "Period",
    "ReadRequest",
    "SearchRequest",
    "SessionInfo",
    "TableResult",
    "TableCatalog",
    "TableDownloadReceipt",
    "TaskTokenIssueRequest",
    "TaskTokenIssueResponse",
    "TrustedTaskTokenIssuer",
    "TrustedTaskTokenProvider",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
