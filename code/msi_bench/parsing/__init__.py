"""The preprocessing pipeline over the real NAVAREA archive (Section 4.3).

Public entry points:

* :func:`msi_bench.parsing.pipeline.parse_message` — parse one raw row.
* :func:`msi_bench.parsing.pipeline.parse_corpus`  — parse the whole archive
  (streamed year-by-year) into a list of :class:`~msi_bench.schema.ParsedMessage`.
* :func:`msi_bench.parsing.chains.build_chains`     — Layer-1 lifecycle chains.
"""

from .pipeline import parse_corpus, parse_message  # noqa: F401
from .chains import build_chains  # noqa: F401
