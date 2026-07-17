# Bundled learned model

`alpakaTune-default.atml` is the optional production artifact used when
`tuning.strategy: learned_hybrid` is selected without an explicit
`learning.model` path. The artifact is produced and validated by the separate
`alpakaTune-ml` repository; raw datasets, training checkpoints, and optimizer
state do not belong in this repository.

A promoted artifact must be accompanied by
`alpakaTune-default.atml.json`, containing its SHA-256 digest, artifact and
feature-schema versions, source training-manifest hashes, supported device
families, held-out-device metrics, license, and provenance. CMake bundles and
installs the model only when the artifact exists, so source checkouts without
an approved model still build normally and the learned strategy uses its
configured fallback.

The production artifact and sidecar together must remain below 5 MiB. Tiny
synthetic artifacts used solely by tests belong under `tests/fixtures`, not
here.
