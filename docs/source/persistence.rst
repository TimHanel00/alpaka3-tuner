History persistence
===================

Each candidate has one in-memory rolling timing history. alpakaTune can persist
that same state in two independent representations: a compact sampled file and
a complete raw-sample file. Both are staged in memory and written once by
their process-wide store during normal shutdown. Kernel launches never open or
append to either JSON file.

Compact history
---------------

``TunerConfig::history`` is the preferred restart source. Its schema contains
only a runtime-ordered selection of measured configurations and, for an active
learned-hybrid strategy, the complete residual-adapter state. Each
configuration record contains its readable values, robust median runtime, and
retained measurement count. It deliberately omits raw timings, execution
counters, policy diagnostics, and timestamps.

``history.sampleCount`` limits the configurations written for each
fingerprint. Omission writes every measured configuration. With a limit of
``N``, the best ``min(N, 3)`` configurations are mandatory. Remaining slots
are drawn without replacement using rank weights declining linearly from 100
for the fastest configuration to 0 for the slowest. The tuner seed and
fingerprint make the draw reproducible for a fixed measured ranking. Selected
records are written in ascending median-runtime order.

Sampling happens only at shutdown write. Same-process readers see every staged
summary. A compact summary is restored as median-valued samples, clamped to the
new tuner's rolling history-window size. This preserves the persisted decision
estimate but cannot reconstruct the discarded runtime distribution.

The compact adapter block duplicates the coefficients, observations, pending
batch count, update count, and model digest stored by complete history. It does
not include model-loading or candidate-pool diagnostics.

Complete history
----------------

``TunerConfig::completeHistory`` owns the former persistence schema. Complete
history schema 11 retains raw rolling samples, sparse Cartesian candidate
arrays, lifecycle and completion state, admission counters, metadata,
timestamps, learned diagnostics, and residual-adapter state. Files created by
the previous ``persistence`` implementation remain structurally compatible
after being moved to the configured complete-history path.

Read precedence
---------------

When both reads are enabled, compact history initializes first. Its timing
summary wins for an overlapping configuration. Complete history then supplies
raw samples only for configurations absent from compact history and restores
its additional lifecycle and diagnostic state. Starting either online mode
then resets run-local sample counts, scheduler state, execution counters, and
completion while retaining the restored timings. Offline mode consumes the
loaded state directly.

A valid compact adapter also wins. Complete adapter state is used only when
compact history has no adapter compatible with the configured model digest and
feature dimensions. Missing or malformed optional adapter data never discards
otherwise valid timing history.

Compact-only ``offline`` mode replays the best summarized configuration.
Compact-only online modes start with those configurations marked as measured
for strategy decisions, with zero current-run samples, execution count, and
terminal completion state.
``loadedFromCache()`` reports true when either source restores compatible
timing or adapter state.

Access and shutdown
-------------------

Each store has its own ``file``, ``read``, and ``write`` controls:

.. list-table::
   :header-rows: 1

   * - ``read``
     - ``write``
     - Behaviour
   * - ``false``
     - ``true``
     - Ignore the old file and replace it at normal shutdown.
   * - ``true``
     - ``false``
     - Load compatible contexts and leave the file byte-for-byte unchanged.
   * - ``true``
     - ``true``
     - Load and merge the newest staged contexts at normal shutdown.
   * - ``false``
     - ``false``
     - Share only process-local staged state.

Omitting a file also makes that store process-local. Compact and complete
history must not name the same file because their schemas are unrelated.

Applications with an explicit end-of-run boundary should call
``alpakaTune::flushPersistence()`` after every thread performing tuned
launches has joined. The call writes each staged store once, reports
serialization and I/O failures to the caller, and is idempotent after a
successful write. Static store destructors provide a fallback for applications
that do not call it; destructor-time failures cannot be reported. Normal return
and ``std::exit`` run those destructors, while abnormal termination cannot
guarantee either write.

Schema and identity
-------------------

Compact history uses file schema 1. Complete history remains file schema 11.
Both use the same workload fingerprint so their contexts can be merged without
storing a Cartesian candidate index in compact configuration records. Compact
values are matched against each named tunable dimension, avoiding enumeration
of the complete candidate space.

The fingerprint includes kernel bundle type, physical device name, additional
identity entries, launch prototype, executor, restrictions, and tunable
definitions. Strategy, execution mode, seed, budgets, history access policy,
sampling count, and learned-model path are excluded.

YAML schema 3 replaces the old ``persistence`` map with the independent
``history`` and ``complete_history`` maps. The old map is rejected. A former
``.alpakaTune/history.json`` full cache can be retained by moving it to
``.alpakaTune/complete-history.json`` before starting with the new defaults.
