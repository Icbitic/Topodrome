# Manual AutoDL deployment

Current operator decision (2026-09-23): the provider API is retired. Rent and
start instances externally, then register their SSH endpoint and password with
`topodrome autodl register`. GPU model/count are inspected rather than selected by
Topodrome. No automatic rental, power-on, replacement, or price/instance-limit
configuration remains.

Beijing B (`beijingDC2`) hosts the existing regional cache at `/root/autodl-fs`.
Preserve its four smaller model caches (Llama-8B, Gemma-12B, OLMo-7B, Qwen-4B)
and the runtime seed. Missing model files use `https://hf-mirror.com`. Other regions
have separate filesystems; registering there requires preparing their cache/secrets.

The uv environment and global mirror cache live on `/root/autodl-tmp`. The verified
runtime seed remains reusable from regional storage; rebuilding Topodrome itself
does not require downloading all weights or dependencies again.

Every new AutoDL bootstrap enables `rclone.verification_via_scheduler: true` for
the Lab1 global mirror. Outputs upload once; Lab1 hashes the stored bytes and
returns only size/SHA-256. Paid workers no longer download their output back just
to verify it. Failed verification still prevents false success.

Rent and start capacity in AutoDL yourself, then register it:

```bash
topodrome autodl register --ssh 'ssh -p PORT root@HOST.seetacloud.com'
# Enter the password at the hidden prompt.
topodrome autodl
```

Stop accepting new jobs and shut down after current work finishes:

```bash
topodrome autodl release --site autodl-manual-EXACT_ID
```

To preempt current jobs and retry them on other eligible sites:

```bash
topodrome autodl release --site autodl-manual-EXACT_ID --force
```

Release requests persist across scheduler restarts. Repeating a request is
idempotent; `--force` may upgrade a pending graceful request. Forced release uses
the existing checkpoint/preemption protocol, retaining the job ID and starting a
new attempt. Work since the last verified checkpoint may be lost; a job without
one starts again. Jobs wait in the queue if no eligible alternative is available.
Explicit site/region affinity remains binding. Interactive kernels cannot migrate:
graceful release waits for them, while forced release stops them.

Release waits for the normal durability barrier and owned-runtime shutdown, then
submits `/usr/bin/shutdown` over SSH. It does not delete an AutoDL rental or shared
storage. The console remains authoritative for billing; an ambiguous SSH shutdown
is surfaced for operator attention. Stopped capacity disappears from active views,
while audit records remain. `autodl unregister --site SITE` is different: it removes
an already-disconnected registration without shutting down its host.

Automatic refresh includes disconnected `starting_failed` and `error` instances,
even if they never sent a heartbeat. `autodl.manual_offline_unregister_seconds`
(default 600) applies after the latest heartbeat, registration, or failure as
applicable. Connected agents, active leases, and nonterminal kernels block removal.
Audit history remains, and power/billing status stays unknown after unregistering.

Idle shutdown remains automatic after five minutes, with required transfers and
active work blocking shutdown. There is no automatic renting, power-on, replacement,
pool limit, provider token, or `autodl on/off` command. Rent and register another
instance when more capacity is needed. GPU type/count comes from inspected inventory.

The Python clients expose `release_autodl(site_id, force=False)` (await for the
async client). The operator-only endpoint is
`POST /api/v1/providers/autodl/registrations/{site_id}/release` with `{"force": false}`.
