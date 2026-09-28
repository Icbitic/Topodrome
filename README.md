# Topodrome

A Python scheduler and execution plane for research experiments across
heterogeneous GPU sites, with a terminal management workbench.

- [Project documentation and setup](topodrome/README.md)
- [User tutorial](topodrome/TUTORIAL.md)
- [Architecture and invariants](TOPODROME_ARCHITECTURE.md)
- [Job arrays](topodrome/docs/JOB_ARRAYS.md)
- [Storage and retention](topodrome/docs/STORAGE_RETENTION.md)

The Python project lives in `topodrome/` and is managed with `uv`:

```bash
cd topodrome
uv sync
uv run pytest tests/unit tests/component
```

Deployment requires site-specific configuration and credentials. Only example
configuration belongs in this repository; model weights, experiment outputs,
credentials, and live service state are excluded. Submitted experiment code is
trusted research code, not a hostile multi-tenant security boundary.
