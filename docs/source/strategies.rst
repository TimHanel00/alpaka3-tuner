Strategies
==========

The queue, execution mode, and strategy have separate responsibilities.
``CandidateQueue`` interleaves active candidates and limits consecutive runs.
The selected :doc:`execution_modes` policy owns admission, measurement
lifetime, revisits, and production launches. A ``ParameterStrategy`` only
recommends a configuration for admission.

Strategy interface
------------------

A parameter configuration is ``std::vector<float>`` with one normalized value
in ``[0, 1]`` per tuning dimension. A strategy receives a read-only
``StrategyContext`` with exactly two operations:

* ``parameterSizes()`` gives the number of discrete values in each dimension;
* ``runtimeFor(configuration)`` returns a read-only ``RuntimeObservation``, if
  that normalized configuration has already been sampled.

The strategy object owns all other state. It does not access the Alpaka queue,
kernel, device, cache, or tuning internals. ``Tuner`` validates the returned
vector, maps it once to the nearest discrete Cartesian candidate, applies the
shared admission gates, and reports the recommendation disposition. A repeated
or rejected point is not silently replaced with another configuration.

The observation exposes the robust runtime estimate, raw and accepted sample
counts, record state, confidence status, and the result of a rank comparison to
the current best record. ``isFinished()`` lets a strategy wait for a stable
result instead of treating a partial configuration as final.

Timing records and comparisons
------------------------------

The record model follows the former ``ConfigRecord``/``MetricContainer``
design. It retains raw samples, tracks raw min/max/mean, and computes a robust
median and mean after MAD-based outlier rejection. The robust median is the
runtime used to select the winner and reported to strategies, so isolated host
scheduling spikes do not distort tuning decisions.

In ``online_fixed``, every ``ci_check_interval`` samples, a legacy-compatible
99% non-parametric
median confidence interval is checked. A record completes once the interval is
within ``ci_relative_width`` and ``minimum_runs_per_candidate`` has been met,
or at ``runs_per_candidate``. If ``mann_whitney_early_stop`` is enabled, the
tuner also compares a non-incumbent record with the current best record after
``mann_whitney_min_samples`` accepted samples each (eight by default). A
statistically slower record is retired early with a directional, one-sided
Mann-Whitney U test at ``mann_whitney_alpha`` (5% by default). The test uses
exact probabilities for small non-tied histories and a continuity- and
tie-corrected normal approximation afterward.

This rank-based retirement is performed before a strategy receives the final
observation, so random, annealing, and Bayesian strategies all operate on the
same statistically filtered history without mutable access to it.

``online_adaptive`` instead ends a candidate residency after its configured
activation burst and maintains a rolling fixed-size sample window. It does not
apply confidence, maximum-sample, or rank-test retirement inside that burst.

Built-in strategies
-------------------

``exhaustive`` enumerates every discrete configuration. ``random`` proposes
uniform normalized points. ``simulated_annealing`` owns an accepted state and
perturbs it with a cooling radius. ``bayesian_optimization`` owns its requested
points and queries their runtimes to fit a bounded RBF surrogate, selecting a
lower-confidence-bound proposal.

``learned_hybrid`` loads a compact offline-trained candidate ranker and scores
a bounded, deterministically sampled candidate pool in batches. It reserves
part of its recommendations for diverse or uncertain points. The shared model
remains frozen while a small residual adapter learns from retired measurements
in the current context and re-sorts only the active pool. Its coefficients,
retained residual observations, partial-batch count, and fit count are stored
with compatible history and restored only for the exact same model artifact.
The core tuner still stores its full legality and runtime-history bookkeeping;
the pool specifically bounds learned inference and learned candidate metadata.
Model
training and campaign data live in the separate ``alpakaTune-ml`` repository;
only a promoted deployment artifact may be bundled here.

Select one in YAML:

.. code-block:: yaml

   tuning:
     strategy: bayesian_optimization
     random_seed: 17

A numeric seed is a deterministic base seed. The tuner mixes it with the
stable tuning-context fingerprint, so the same context is reproducible while
different kernels and identity entries do not receive identical random or
learned candidate streams. Use ``random_seed: nondeterministic`` only when a
fresh, non-reproducible stream is explicitly desired.

Custom strategies derive from ``alpakaTune::ParameterStrategy`` and implement
``recommend(StrategyContext const&)``. The only required output is a valid
normalized vector. They may override ``recommendationResult`` when their state
must react to the shared tuner's admission result.
