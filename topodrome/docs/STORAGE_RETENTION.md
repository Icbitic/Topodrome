# Storage lifecycle

Lab1 remains authoritative. HTTP remains a transport, not the authority for
deletion. Retention policy and references decide which owned files can expire.
See the tutorial's authoritative-storage retention section for settings and
commands. The legacy retention rollout needed no database revision; switching
to the blob layout requires a coordinated scheduler/site-agent upgrade.

## Immutable blob layout

With `storage_layout: blobs-v1` on the scheduler and every participating agent,
new completed outputs and uploaded asset files live at
`blobs/sha256/<first-two-hex>/<sha256>`. Blobs are whole files, not fixed-sized
chunks. Schema-2 manifests retain logical names, sizes and hashes; permissions
remain on job and asset metadata, never on possession of a digest.

Publication verifies bytes, installs immutable payloads atomically, then commits
the manifest. Job outputs register durable pending claims before transferring
bytes. The authority-local `.blob-index/index.sqlite3` records owned objects and
claims; a cross-process lock coordinates publication and collection. PostgreSQL
state and asset manifests determine lifetime. Existing verified blobs are reused;
publishing an output as an asset adds a reference without another payload copy.
Direct client uploads still send their contents before server-side deduplication.

Collection preserves published assets, accepted consumers, retained artifacts,
live leases, active downloads, and pending output claims. It enumerates indexed blobs, rechecks
references under the publication lock, and deletes only unreferenced objects
past the grace period. Unknown or malformed metadata fails closed. Checkpoints
and logs keep their separate formats and retention rules. Native and HTTP
mirror storage use same-filesystem hardlinks for atomic local installation.

## Temporary download protection

The unified `download` command resolves `job://` or `asset://` members under the
same reference lock as publication and GC. Fsynced owner-bound descriptors in
`.topodrome-downloads/download_<id>.json` pin those exact files without publishing
an asset or changing their ordinary retention. GC and its eligibility forecast
exclude live download pins. Invalid pin metadata aborts collection.

Pins persist across scheduler restarts, expire after 15 minutes without renewal,
and cannot be renewed past 24 hours. Clients and active server streams renew
every minute; clients close pins on completion/failure, and abandoned pins expire
after process death. The next download prunes expired descriptors. Capacity is
bounded to 16 live pins per owner / 128 overall; each admits at most 10,000 files
and 16 MiB of metadata. No network transfer holds the storage reference lock.
An already-open descriptor remains readable if a caller closes its pin early,
but later member requests require a live pin. Local failed-download partials
are outside authoritative storage and remain for explicit retry.

### Fresh-start archive (no migration)

`scripts/archive_storage_fresh.py` defaults to read-only inventory. Applying it
requires the scheduler and agents to be stopped and every job/kernel/lease to
be terminal. It backs up PostgreSQL and actually restores the dump into a
disposable database to verify inventory before moving any data. It journals
same-filesystem renames of the old jobs, assets, projects and log/retention
trees, then clears job/kernel metadata transactionally. Users, sites, model
configuration and GPU reservations remain. An explicit layout marker rejects
new agents configured for the wrong layout. Offline agents must be upgraded
before being re-enabled.

The archive is recoverable offline, not exposed by active job/asset APIs and
not eligible for active GC. Old references do not migrate. On failure services
remain stopped: inspect the file journal and database dump before recovering;
never start a mixed-layout deployment. Rollback requires restoring the matching
database, trees, code and configurations together while idle. Do not restore an
old database over newly accepted work. A same-disk archive is not a backup
against disk loss and does not free the archived bytes.

### Blob rollout acceptance, 2026-09-27

- 1,119 unit/component tests passed; changed-module lint and user-skill
  validation passed. Tests exercise deduplication, canonical manifest paths,
  claim-before-upload ordering, retained/active input pins, private access,
  corrupted-object rejection, archive boundaries and layout mismatch rejection.
- Scheduler and Lab1/SQZ agents use `blobs-v1`. Jupyter remains offline and was
  not upgraded. No AutoDL capacity was rented. Its registration bundle was
  rebuilt with a portable frozen-lock-validated runtime seed, repairing a
  previously missing seed reference.
- Old storage is at
  `/ssd/xiangkun/.local/share/topodrome/archive-before-blobs-20260927`;
  its sibling `archive-before-blobs-20260927-metadata` contains the PostgreSQL
  dump and pre-cutover configurations. The dump was restored into a disposable
  database and its inventory checked before cutover. All 907 old jobs and 583
  asset manifests were archived, not migrated. All 3,345 asset members remain
  present with recorded sizes, and catalog metadata hashes are unchanged.
- Real Lab1 producer `job_31b9e634b5e1455997b9d10de043ea3d` and SQZ consumer
  `job_1976ff6e39384a9b8aa4c824f2ad41cc` succeeded. Publication, download,
  direct upload, remote input staging and a new SQZ output upload passed.
  Identical output from both jobs resolved to one canonical blob with one link.
  Acceptance records have their own
  `/ssd/xiangkun/.local/share/topodrome/archive-cas-acceptance-20260927` archive
  (and sibling `-metadata` backup) so production starts empty.
- The old 21 site records and 58 reservation records are preserved; user
  credentials and model configuration were not changed. Archive storage is
  outside active GC and remains subject to explicit operator recovery/removal.

## Safety boundaries

### Asset lifecycle rollout — 2026-09-27

Deletion and editable tags were deployed to the scheduler while no jobs or
kernels were active; GPU agents were not restarted. The hermetic suite passed
1,274 tests, changed-module lint and shared-skill validation passed, and live
tag add/filter/remove and idempotent deletion checks passed.

Only the six previously identified private `indexed-retrieval-eot-template`
references were removed, with versions `job_fbca24b1c7634e91b8199009cb49dd24`,
`job_6c939d41df4c493882f07f50af03dce9`, `job_cc450b40548e43bcac6867195b682368`,
`job_c4eba83a26c1468693e2b07205d2dc0f`, `job_ae5170f9f9f143ad85ad0975ef4f5c63`,
and `job_0849338c2a2c4cbda7dcbc35153717fe`. Their six succeeded producer jobs and
18 payload files remained unchanged. Nine newer references were preserved;
no immediate payload collection was requested. Deleted refs are tombstoned,
and their payloads remain subject to the ordinary reference/retention rules.

### Collection invariants

- Only database-known terminal jobs and terminal attempts without live leases
  are eligible. Pending finalization and recovery remain protected.
- Every published asset manifest pins its members, regardless of version,
  owner or visibility. Active consumers and their restore references add pins.
- Asset deletion removes only the selected catalog reference under the same
  storage lock used for publication, input pinning and collection. A fsynced
  `deleted.json` audit tombstone precedes unlinking `manifest.json`; readers hide
  tombstoned versions, and a repeated deletion completes an interrupted unlink.
  A leftover manifest conservatively pins bytes until then. Tombstones and
  mutable `tags.json` metadata do not pin payloads. Reusing deleted name/version
  pairs is forbidden. Existing consumers, other assets and ordinary artifact
  retention remain effective. There is no immediate payload wipe or undelete;
  after normal GC, audit metadata cannot recover collected bytes.
- Only registered committed artifacts, writer-format recovery checkpoints,
  and terminal-attempt staging files are candidates. Logs and unknown
  committed files are not deleted.
- Candidate discovery, submission pinning, publication, and deletion share a lock across
  scheduler processes. Thread cancellation waits for filesystem work before
  releasing the lock. Collection uses bounded passes and a durable audit.
- Each file is checked for symlinks, size/identity changes, and grace-period
  age. Malformed/unreadable catalog data defers collection rather than risking
  deletion. Cleanup removes no directories recursively.

## Rollout and rollback

Run unit/component tests, inspect a dry-run, and inventory asset manifests and
member sizes before enabling automatic collection. Upgrade the scheduler only
while idle; do not restart GPU agents. Verify the API dry-run before applying
bounded cleanup passes, then compare the asset inventory and filesystem usage.

Disable `storage_retention.enabled` and restart the idle scheduler to stop
automatic cleanup. Rollback cannot restore deleted payloads; journals are audit
metadata, not a recycle bin. Published assets and active work must remain intact
throughout. Do not hand-delete attempt trees to reclaim space.

## Deferred stages

Old-data migration is intentionally not provided. Log
compression/expiry, source/orphan-upload cleanup, and remote output-spool GC need
their own migrations and safety tests. The first rollout deliberately keeps
the existing file layout and never expires remote pending finalization by age.

## Second stage: bounded live checkpoints

`checkpoint_keep_snapshots` defaults to two complete snapshots per live attempt.
The writer rotates only after successful atomic save/fsync/hash. Before selecting
a snapshot for verification, the backend fsyncs an exact filename pin beneath
`.retention-pins`, serialized against rotation by `.retention.lock`. Pins are
never dropped by live rotation, even after failed verification. Other attempts'
directories (including restore inputs) are never traversed. No lock spans
network I/O; rotation skips on contention. At most 100 files are removed per
save, so accumulated legacy history drains gradually rather than stalling one
save. This is not a strict disk quota: pins, hardlinks and failures can exceed
the configured count. Terminal-job cleanup still owns later authoritative
expiration. Small lock/pin metadata is left in the attempt directory.

The worker's in-memory checkpoint history also keeps only two records, avoiding
unbounded references to old model/optimizer payloads. Tests cover persistence
failure, verification/rotation exclusion, pin survival across restart, unsafe
paths, numeric sequence ordering, configurable retention and heap reclamation.
Remote output-spool GC and source GC remain deferred.

Second-stage acceptance, 2026-09-27: 1,106 unit/component tests passed, lint and
skill validation passed. Lab1 and SQZ agents were upgraded with the scheduler
idle gate; no GPU workload was submitted or interrupted. Both sites returned
ready. SQZ's initial bootstrap failed during SSH-forward setup; explicitly
ensuring the configured owned forward succeeded and startup then returned ready.
Filesystem and deployed-writer smoke tests passed on Lab1/SQZ (two newest
snapshots plus a pinned fallback), and the five changed production modules
matched across the sites. Jupyter was left offline and untouched. Existing
workload payloads were not bulk-pruned during this stage; rotation applies on
future saves. SQZ's previous source package was retained at
`/cephfs/tianhaolin/sxk/topodrome/release/src/topodrome.before-nonblocking-7cd75f82026b`.

## Lab1 acceptance, 2026-09-27

- Hermetic suite: 1,097 passed; lint and user-skill validation passed. The suite
  uses umask 022 because existing permission tests construct public directories.
- Restarted only the idle scheduler; Lab1's agent was not restarted. Readiness
  and authenticated status returned 200; Lab1 and SQZ reported ready. Jupyter
  remained offline and was not modified by this rollout.
- API dry-run matched the prior read-only inventory. Eighteen bounded deletion
  passes removed 13,246 files / 347,245,953,024 allocated bytes (323.4 GiB);
  a subsequent pass found zero candidates. All candidates belonged to
  `@operator`; no other owner's files were selected.
- All 583 published manifests were unchanged, and all 3,345 unique referenced
  files remained present with their recorded sizes (77,048,770,249 bytes total).
  Payload checksums were not reread. Catalog metadata inventory SHA-256:
  `312d27d9d905cdc6243314547f4cc92b51b7ffae402816317b66c8c2bc3e9848`.
- Global storage decreased from about 438 GiB to 114 GiB. Filesystem available
  space increased from about 477 GiB to 801 GiB. Automatic collection is enabled
  every ten minutes with the documented defaults. Exact permanent deletions
  are recorded in the retention journal, not backed up there.
