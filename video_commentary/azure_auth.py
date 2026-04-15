"""Azure authentication helpers — API-key and Managed Identity (MSI).

When an API-key environment variable is set the corresponding service uses
key-based authentication (the current default).  When the key is *absent*,
the helpers transparently fall back to ``DefaultAzureCredential`` from the
``azure-identity`` package, which supports Managed Identity, Azure CLI,
environment-based credentials, and more.

Install the optional dependency once::

    pip install visual-commentary-pipeline[azure]
"""

from __future__ import annotations

import threading
from typing import Any

_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

_credential: Any = None
_credential_lock = threading.Lock()


def _get_credential() -> Any:
    """Return a cached ``DefaultAzureCredential`` instance."""
    global _credential
    if _credential is None:
        with _credential_lock:
            if _credential is None:
                try:
                    from azure.identity import DefaultAzureCredential
                except ImportError:
                    raise SystemExit(
                        "No API key configured and azure-identity is not installed.\n"
                        "Either set the relevant API key environment variable, or "
                        "install azure-identity for Managed Identity auth:\n"
                        "  pip install visual-commentary-pipeline[azure]"
                    )
                _credential = DefaultAzureCredential()
    return _credential


def get_bearer_token() -> str:
    """Acquire a Bearer token for Azure Cognitive Services via ``DefaultAzureCredential``."""
    return _get_credential().get_token(_COGNITIVE_SCOPE).token


def azure_openai_auth_headers(api_key: str | None = None) -> dict[str, str]:
    """Return auth headers for Azure OpenAI.

    * If *api_key* is provided → ``{"api-key": api_key}``
    * Otherwise → ``{"Authorization": "Bearer <token>"}`` via MSI
    """
    if api_key:
        return {"api-key": api_key}
    return {"Authorization": f"Bearer {get_bearer_token()}"}


def cognitive_services_auth_headers(api_key: str | None = None) -> dict[str, str]:
    """Return auth headers for Azure Cognitive Services.

    Covers Speech, Content Understanding, and Document Intelligence.

    * If *api_key* is provided → ``{"Ocp-Apim-Subscription-Key": api_key}``
    * Otherwise → ``{"Authorization": "Bearer <token>"}`` via MSI
    """
    if api_key:
        return {"Ocp-Apim-Subscription-Key": api_key}
    return {"Authorization": f"Bearer {get_bearer_token()}"}
