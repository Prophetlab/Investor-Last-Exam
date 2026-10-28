"""Explicit operator choices, separate from task semantics and customer ZIP layout."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeploymentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    schema_version: Literal["deployment.v1"] = "deployment.v1"
    backend: Literal["docker", "e2b"]
    docker_network: Literal["harbor", "none"] = "harbor"
    agent_image: str | None = Field(default=None, min_length=1)
    # A separate verifier image can embed task-specific private inputs. It must
    # not be applied to every task by an implicit global image override.
    verifier_images: dict[str, str] = Field(default_factory=dict)
    concurrency: int = Field(default=1, ge=1)
    attempts: int = Field(default=1, ge=1)
    lease_seconds: int = Field(default=1800, ge=60, le=14400)

    @model_validator(mode="after")
    def validate_backend_options(self):
        if self.backend != "docker" and self.docker_network != "harbor":
            raise ValueError("docker_network only applies to Docker")
        if any(not value.strip() for value in self.verifier_images.values()):
            raise ValueError("verifier image references cannot be empty")
        return self


def example_profiles() -> dict[str, dict]:
    """No customer, host, image registry or API key is presumed."""
    return {
        "e2b.example.json": {
            "schema_version": "deployment.v1",
            "backend": "e2b",
            "concurrency": 1,
        },
        "docker.example.json": {
            "schema_version": "deployment.v1",
            "backend": "docker",
            "docker_network": "harbor",
            "concurrency": 1,
        },
        "docker-none.example.json": {
            "schema_version": "deployment.v1",
            "backend": "docker",
            "docker_network": "none",
            "concurrency": 1,
        },
    }
