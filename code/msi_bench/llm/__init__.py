"""Provider-agnostic LLM clients for synthetic data generation (Section 4.3).

Three families are configured (``config.yaml``): generators A and B,
and judge C. Each is a :class:`LLMClient`. When ``llm.enabled`` is false, a key
is missing, or the provider is ``mock``, a deterministic :class:`MockClient` is
returned so the *entire* benchmark + experiment pipeline runs end-to-end with no
API access (channel separation means only the narrative-text channel is even
affected). Configure real keys in ``.env`` to produce publication-grade text.
"""

from .client import LLMClient, MockClient, get_client, get_generator_clients  # noqa: F401
