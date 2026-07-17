Persistent tuner cache
======================

After a tuner finishes, alpakaTune writes a versioned JSON record to
``TunerConfig::persistenceFile``. The record contains the winner, raw candidate
timings, robust per-candidate estimates, readable tuning-parameter values,
completion reason, limits, tuner metadata, and timestamps. The explicit
candidate mapping lets analysis tools report configurations instead of only
Cartesian-space indices. The record also
records a milestone whenever a retired configuration establishes a lower
robust runtime than every configuration retired before it. Each milestone
contains the candidate, total launch count, retired-configuration count,
runtime, and elapsed wall time. A later tuner with the same fingerprint
immediately enqueues the cached winner without an additional timing
synchronization.

History schema 10 also contains ``metadata.model_context``. It provides a
strategy-independent workload identifier, CPU/GPU device class, ordered typed
tuning dimensions, concrete numeric values where representable, and automatic
numeric/hash features for the kernel, launch, context, and device. Offline
tooling uses this structured record instead of parsing printable configuration
strings. Learned runs additionally record artifact loading/fallback status and
online-adapter progress.

After every measured launch, the process-wide persistence store retains an
in-memory snapshot. If the program returns normally or calls ``std::exit``
before a tuner completes, static destruction writes the latest snapshot with
``completion_reason`` set to ``none``. Such an incomplete record remains
available for benchmark analysis but is not reused as a cached winner on the
next run. A shutdown snapshot never replaces an already completed record for
the same tuner fingerprint.

The fingerprint includes the kernel bundle type, physical device name,
additional tuner identity entries, configuration values, frame prototype, marker
layout, and all tunable definitions. Changing any of these creates a new cache
record and starts tuning again. Learned histories also include the selected
model artifact content digest in their identity, so replacing an artifact at
the same path cannot silently reuse a result from another model. Remove a
cache file to force a retune.
