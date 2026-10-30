from __future__ import annotations


class DataLakeError(Exception):
    """Base class for expected data-lake failures."""


class ConfigurationError(DataLakeError):
    """Deployment configuration is incomplete or unsafe."""


class TimeNormalizationError(DataLakeError, ValueError):
    """A timestamp cannot be normalized without risking future leakage."""


UNTRUSTED_REVISION_TIME_CODE = "UntrustedRevisionTime"


class AuthorizationError(DataLakeError):
    """A principal is not authorized for an operation."""


class DataLakeClientError(DataLakeError):
    """A Tool API client request failed without exposing credentials."""


class DocumentUnavailableError(DataLakeError):
    """A document is absent or not visible to the caller."""


class SearchScopeError(DataLakeError):
    """A metadata search is too broad to execute safely."""


class SearchCursorError(DataLakeError):
    """A search continuation cursor is invalid or out of scope."""


class ObjectStoreError(DataLakeError):
    """An object-store operation failed its contract."""


class ObjectAlreadyExistsError(ObjectStoreError):
    """An immutable object key already exists with different content."""


class ObjectNotFoundError(ObjectStoreError):
    """An object key does not exist."""


class StorageRoleDisabledError(ObjectStoreError):
    """A logical storage role is disabled in this deployment."""


class IngestionError(DataLakeError):
    """An ingestion job or artifact violates the commit protocol."""


class IngestionSliceYield(IngestionError):
    """A connector saved its frontier and yielded without a discovery failure."""


SOURCE_AUTHENTICATION_EXIT_CODE = 78


class SourceAuthenticationError(IngestionError):
    """A source requires credential, MFA, or account recovery before retrying."""


class PermanentIngestionError(IngestionError):
    """A source job cannot succeed on retry and requires explicit evidence."""


class SourceRecordSupersededError(PermanentIngestionError):
    """A source withdrew a record and exposes a replacement for its lineage."""


class SourceVersionUnavailableError(IngestionError):
    """A mutable source no longer exposes an exact version seen at discovery."""

    def __init__(
        self, message: str, *, evidence: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.evidence = evidence


class LeaseConflictError(IngestionError):
    """A worker attempted to mutate a job leased by another worker."""
