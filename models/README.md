# Bundled learned model

`alpakaTune-default.atml` is the optional bundled model used when
`tuning.strategy: learned_hybrid` is selected without an explicit
`learning.model` path. Model training and validation tools are in the separate
`alpakaTune-ml` repository. You can also set `learning.model` to an external
artifact.

## Packaging a model

Include a metadata sidecar,
`alpakaTune-default.atml.json`, containing its SHA-256 digest, artifact and
feature-schema versions, source training-manifest hashes, supported device
families, held-out-device metrics, license, and provenance. CMake bundles and
installs the model only when the artifact exists. Without an available model,
the learned strategy uses its configured fallback.

The bundled artifact and sidecar together must not exceed 5 MiB. Synthetic
models for tests are stored under `tests/fixtures`.
