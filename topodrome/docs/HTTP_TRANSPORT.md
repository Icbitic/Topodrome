# HTTP-only experiment transport

Lab1 is the sole storage authority. Every remote experiment-data operation uses
the scheduler's authenticated, enrolled-agent HTTP API: metadata, listings,
tiny files, bulk inputs, outputs, and runtime seeds. Lab1 itself uses native
filesystem access. There is one remote protocol and no fallback backend.

## Persistent connections, isolated capacity

Metadata/tiny requests and bulk streams use separate bounded HTTP connection
pools and server I/O capacity. Tiny files default to at most 1 MiB (configurable
up to 4 MiB; zero selects streaming). A small GET verifies the content SHA-256
and atomically publishes a fsynced local file. A small PUT combines upload,
authority verification, and durable acknowledgement. Small immutable source
trees use one bounded ZIP wire envelope, verified against their original
manifest; the envelope does not become another stored source identity.

If a site uses SSH tunneling, configure two independently supervised forwards:
`ssh_local_forward` for control and `bulk_ssh_local_forward` for bulk, with
separate local ports/state directories and `storage.bulk_url`. Both reach the
same authority, pin host keys, and disable ControlMaster reuse. Multiple HTTP
pools on one outer SSH TCP connection would still share loss/congestion.
The agent checks both owned routes every five seconds; a failed bulk route
does not tear down healthy control. Exited owned children are reaped before
checking process-group liveness. Recovery can also recognize a proven zombie
left by a previous supervisor, but refuses replacement if any live group member
remains; it never signals an unproven process. Authentication appears only in headers.
Non-loopback HTTP origins must use TLS. No extra public storage daemon is needed.

### Measured renewal of slow, live bulk routes

An optional `storage.bulk_reconnect` policy uses two explicitly configured
loopback slots for the same pinned SSH authority. It is disabled by default.
No Mac proxy, new relay, provider operation, or alternate storage authority is
introduced. Example for a site whose bulk/control ports are 18043/18042:

```yaml
storage:
  bulk_url: http://127.0.0.1:18043
  bulk_reconnect:
    enabled: true
    candidate_port: 18044
    slow_mib_per_second: 1
    sustained_seconds: 30
    cooldown_seconds: 600
    min_speedup: 1.5
    probe_timeout_seconds: 5
```

Only sustained active bulk traffic below the threshold triggers a comparison;
idle periods do not. A background worker establishes the candidate first, then
compares two bounded, checksummed 128-KiB samples in each direction over old and
fresh SSH connections. Probe data stays in memory; it is not an artifact upload
or disk-speed benchmark. A candidate must improve the weaker direction by at
least 50% without regressing either direction by more than 20%. An unavailable
old link may be replaced only by a successfully verified candidate. Both
successful and failed attempts observe the ten-minute cooldown. Unsupported
authorities, backpressure and failed candidates leave the current route alone;
authentication and integrity failures are never accepted as speed samples.

New HTTP requests use the accepted generation. Open streams retain their old
pools/tunnel until they drain; an upload can change generations between durable
extents, and a failed download retains its independently verified resume
prefix. At most two owned generations exist, so a draining stream cannot cause
an unbounded reconnect storm. Control traffic, actors, and running computation
are never restarted by this policy. Rollout still requires the normal idle and
outbox deployment gates below.

A fresh SSH TCP flow can select a different flow-hashed network path, but there
is no guarantee of a route change. Routers can also reroute existing flows.
Measured renewal avoids switching merely because a connection is old; it does
not fix persistent physical loss, overloaded disks, or a uniformly bad path.

## Resumption, integrity, publication

Large downloads stream into private contiguous partials, with bounded memory.
Range reads validate Content-Range, exact length, and If-Match source identity.
The metadata ETag is an opaque filesystem version, not a claimed content hash.
Input pins or Lab1's independently observed SHA-256 verify the completed file
before atomic exposure. Changed versions, symlinks, special files, and corrupt
bytes fail closed. HTTP listings are paged and bounded. Tree members verify
before one atomic directory publication; a cached child never implies a
complete parent tree. Tree staging/partials use a separate private namespace.

Reading an immutable tree and preparing a writable attempt are separate
operations. `resolve` may return a verified directory-cache view;
`resolve_writable` always returns the canonical local path, without fetching
remote history. Logs, checkpoints, staging and manifest writers use canonical
paths, which are also the sources of authoritative upload. New attempts and
upload-only restart therefore cannot write inside an older downloaded job tree.
Symlink aliases and special files are refused for write locations. Missing
local files and other untyped filesystem failures are surfaced as diagnostics,
not retried forever as network failures; recoverable transport outages retain
the existing upload-only backoff and durable receipts.

Uploads journal source size/hash and the exact owned private prefix before
sending. Checksummed extents start at 128 KiB and adapt toward about two seconds
per ACK, capped at 8 MiB. Each ACK follows authority file/directory fsync. Lost
ACKs cannot duplicate bytes or create holes. On restart the stored prefix is
hashed and compared with the local source; length alone is not trusted.
The complete private file verifies before atomic publication, and Lab1's final
file verifies before the journal retires. A lost publication ACK is recovered
without resending the completed file. Payload conflicts fail; only the current
manifest pointer is replaceable.

Upload extents optionally use HTTP `Content-Encoding: gzip`, negotiated through
the authority's `Accept-Encoding` response header on existing prefix queries or
acknowledgements. No discovery round trip is added; older clients/authorities
continue to exchange identity bodies. Compression uses a cheap bounded sample,
then level-1 gzip only when it saves at least 10% and 4 KiB. Tiny and already
compressed extents stay raw. Each body is independently encoded; headers and
credentials never enter a compression context.

Wire length and decoded length are both bounded to 8 MiB. Decoding rejects
expansion past the declared extent, truncated/corrupt trailers, trailing bytes,
and concatenated members before any prefix write. Offsets, SHA-256, durable ACKs,
progress, resume receipts and stored blobs always describe the original bytes.
Compression never changes a content identity or storage layout. Unsupported or
corrupt encodings fail closed, never trigger an unverified raw retry.

Prefix inspection/publication has a separate four-slot I/O limit from the two
body-reception slots. A slow upload therefore cannot consume all status-check
capacity. Exact-prefix filesystem locks still fence simultaneous modifications.
Disconnected or timed-out bodies release their reception slot without writing
an incomplete extent.

A total transfer deadline covers at most three link-recovery attempts/backoff.
Explicit 429/503 Retry-After backpressure has a separate cancellable two-minute
budget. Authentication, integrity, disk errors, cancellation, and expired
publication authority never trigger a protocol downgrade or blind retry.
Protected pending outputs survive failure for recovery/retention. Progress is
not a durability acknowledgement. Files still being changed by a workload are
not uploaded as completed outputs. Model caches remain independently managed.

## Configuration and administration

Use `configs/site.autodl.example.yaml` or `configs/site.jupyter.example.yaml`.
Lab1 leaves `storage` unset. Remote agents configure `storage`, an authority ID,
and the optional separate bulk route. Unsupported YAML fields fail explicitly.
Preserve an existing opaque `storage.authority_id` when upgrading a site with
pending receipts. No content layout or database migration is needed.

Upgrade the scheduler first. Before restarting any agent, check jobs, leases,
kernels, input staging, pending finalization, and event/log outboxes. Never
restart underneath computation or undelivered finalization.

AutoDL bundles contain uv and Topodrome source. A stdlib-only helper uses
protected enrollment to fetch/resume the pinned runtime seed over HTTP before
dependencies are installed. Its independently owned bulk tunnel stops after
acquisition; the long-lived agent then owns the control and bulk routes.

One-time administrative installation streams files through SSH stdin to a
stdlib Python receiver. It confines writes to an exact private owned directory,
refuses symlink parents/targets, verifies length/SHA-256, fsyncs, and atomically
publishes each member. The release pointer is committed last. An interrupted
installation preserves the previous release. This administration path is not
used for experiment data and needs no remote transfer helper package.

## Validation and limits

The hermetic tests use real HTTP sockets for transfers exceeding 2 GiB in both
directions, cancellation and restart with nonzero resume offsets, corruption,
lost chunk/publication ACKs, unsafe paths, paged trees, filename collisions,
bounded backpressure, and separate tunnel ownership. SSH installation also has
real-channel coverage plus truncated/corrupted payload and symlink tests.

The 2026-10-01/02 rollout passed all 1,523 unit/component tests after removal of
retired adapters, scripts, configuration and protocol-specific tests. Lab1 and
SQZ passed fresh idle/outbox gates before activation. Both services were healthy
after deployment; the manual AutoDL bootstrap was rebuilt using the existing
verified runtime seed, without renting capacity or downloading model weights.
All Lab1 test data, reports, deployment staging and bulk caches were on `/ssd`.

The GPU-free live SQZ test used a 2,097,169-byte fixture. It resumed a real
broken-tunnel download from a 1,081,344-byte prefix, and an interrupted upload
from its 131,072-byte durable ACK. Lab1 independently checked final length and
SHA-256. The resumed download/upload took 17.46/35.64 seconds respectively.
Five uncached 40-byte requests over persistent HTTP had a 0.75-second median
and 4.35-second maximum: this is a variable-link sample, not a latency guarantee.
A separate owned-bulk-tunnel fault injection recovered in 4.99 seconds while
the control tunnel generation remained unchanged. Only private smoke fixtures
were removed after verification; user outputs and model caches were untouched.
There was no live paid AutoDL test, and offline Jupyter was not upgraded.

The 2026-10-07 writable-workspace fix passed all 1,665 unit/component tests,
including live-authority HTTP tests for retries after an older attempt's tree
was cached and upload-only restart without scratch. An affected SQZ2 attempt
completed without rerunning computation: its original final log was recovered
to the canonical path, then uploaded and acknowledged through the normal
agent protocol. Its five output blobs and final log were independently verified
on Lab1; no receipt or job state was manually bypassed. Independent idle-gated
agent upgrades completed on SQZ2, SQZ and Lab1 with fresh healthy heartbeats.
The scheduler remained running. The manual AutoDL bootstrap was rebuilt and
verified using the existing runtime seed; no paid instance or model download
was needed, and the offline Jupyter installation was left unchanged.

Persistent connections remove process/handshake startup, not WAN RTT, packet
loss or storage contention. No multi-GiB WAN throughput claim is made. SSH
management reachability and an agent's outbound data route are independent.

Read-only 2026-10-02 path tests showed that both sites downloaded 8-MiB model
ranges through HF-Mirror at several MiB/s while direct SQZ/Lab1 transfers suffered
heavy retransmission and much lower throughput. Isolated `IPQoS=none` and
1,200-byte MSS tests did not cure the problem, so no speculative host-wide or
production SSH changes were applied. A temporary relay route was only modestly
better and was closed after testing; production does not acquire a new Mac
dependency. Upload compression mitigates compressible output, not the route's
physical loss or already-compressed data. Replaying the affected emoji job's
13,864,335 bytes through the extent encoder yielded 642,140 wire bytes (~21.6x
less) in a local CPU test; this is not an end-to-end completion-time guarantee.
