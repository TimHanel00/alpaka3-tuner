Persistent context cache
========================

After a context finishes, alpakaTune writes a versioned JSON record in the
YAML ``persistence.directory``. The record contains the winner and measured
candidate timings. A later context with the same fingerprint immediately
enqueues the cached winner without an additional timing synchronization.

The fingerprint includes the kernel bundle type, physical device name,
additional context identity entries, session defaults, frame prototype, marker
layout, and all tunable definitions. Changing any of these creates a new cache
record and starts tuning again. Remove a cache file to force a retune.
