Persistent tuner cache
======================

alpakaTune always stages versioned records in memory. A persistence file is
optional. When ``TunerConfig::persistenceWrite`` is enabled and
``TunerConfig::persistenceFile`` is set, the process-wide persistence-store
destructor writes the newest staged records once during normal process
shutdown. There is no per-launch file open, JSON serialization, or append.
``std::exit`` also runs the static destructor; abnormal process termination
cannot guarantee a flush.

Reading and writing are independent. ``persistenceRead = false`` prevents any
existing file from being loaded. If writing is enabled at the same time, the
shutdown write starts a fresh file and replaces the old one instead of merging
it. ``persistenceRead = true`` with ``persistenceWrite = false`` loads history
for ``offline`` replay or ``onlineAdaptive`` context while leaving the file
byte-for-byte unchanged. Omitting ``persistenceFile`` provides only the shared
in-process store and never touches the filesystem.

The record contains the current best candidate, rolling candidate timings,
robust per-candidate estimates, readable values for measured tuning-parameter
configurations, execution
mode and admission counters, completion reason, limits, tuner metadata, and
timestamps. The explicit candidate mapping lets analysis tools report
configurations instead of only Cartesian-space indices. A milestone is also
recorded whenever a retired configuration establishes a lower robust runtime
than every configuration retired before it. Each milestone contains the
candidate, total launch count, retired-configuration count, runtime, and
elapsed wall time.

Candidate arrays retain one stable position per Cartesian index, but unmeasured
configuration entries remain ``null``. This keeps adaptive startup proportional
to measured history instead of eagerly printing the entire Cartesian space.

Mode-specific reuse
-------------------

A later ``offline`` tuner with the same fingerprint immediately enqueues the
best loaded or in-process-staged configuration without timing or
synchronization.
``online_fixed`` resumes an incomplete record or replays a completed winner.
``online_adaptive`` resumes either kind of compatible measured history and
continues adapting. Its cumulative execution count is retained for provenance,
but each new tuner process receives a full new adaptive horizon. The
history-aware horizon offset controls only the initial normalized
sigmoid/Boltzmann progress.

An adaptive or interrupted record normally has ``completion_reason`` set to
``none`` and is still valid input to ``offline`` and ``online_adaptive``. A
newer shutdown snapshot replaces an older record for the same fingerprint,
including a formerly completed fixed-mode record. This is necessary because
adaptive runs continue updating rolling samples after a finite experiment may
previously have selected a winner.

Schema and identity
-------------------

History schema 11 contains ``metadata.model_context``. It provides a
strategy-independent workload identifier, CPU/GPU device class, ordered typed
tuning dimensions, concrete numeric values where representable, and automatic
numeric/hash features for the kernel, launch, context, and device. Offline
tooling uses this structured record instead of parsing printable configuration
strings. Learned runs additionally record artifact loading/fallback status,
artifact digest, bounded-pool diagnostics, and the complete lightweight
residual-adapter state: fitted coefficients, retained residual observations,
pending batch progress, and fit count. A later learned run restores that state
only when the model artifact digest and tuner fingerprint remain compatible.
Missing, malformed, or model-incompatible adapter state is ignored without
discarding otherwise valid timing history.

The persistence fingerprint includes the kernel bundle type, physical device
name, additional tuner identity entries, launch prototype, marker layout,
restrictions, and all tunable definitions. Strategy, execution mode, random
seed, budgets, sampling policy, and learned-model path are deliberately not
part of it. A compatible history can therefore cross strategy and mode
boundaries; metadata still reports which policy produced its newest snapshot.
Changing a workload or device identity creates a new record.

Schema 11 is intentionally a fresh-history boundary. Earlier persistence
schemas are rejected rather than migrated because their lifecycle and sample
retention semantics are ambiguous under the three execution modes. Remove or
archive an old cache file before collecting new histories.
