# Topodrome contributor rules

For using the deployed system rather than changing its implementation, read
`TUTORIAL.md` first.

Read `../TOPODROME_ARCHITECTURE.md` before changing behavior. Preserve all
invariants in section 17. In particular, physical GPU admission belongs to the
site agent, artifact completion requires authoritative verification, provider
operations are journaled before mutation, and credentials never enter job
payloads or logs.

Use `uv` for environments, dependency changes, and commands. Run the hermetic
suite before external integration tests:

```bash
uv run pytest tests/unit tests/component
```

Paid/provider tests require an explicit opt-in and must power off test capacity
in a `finally` path.
