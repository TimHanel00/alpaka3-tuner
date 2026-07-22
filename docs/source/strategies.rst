Strategies
==========

The queue and the strategy have separate responsibilities. ``CandidateQueue``
interleaves active candidates and limits consecutive runs. Each candidate has
its own record lifecycle: activation warm-up, measurement, confidence or
maximum-run completion, then retirement. A ``ParameterStrategy`` only
recommends which new configuration should enter that queue.

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
vector, maps it to the nearest discrete Cartesian candidate, and selects the
nearest unscheduled candidate if the strategy repeats a point.

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

Every ``ci_check_interval`` samples, a legacy-compatible 99% non-parametric
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

Built-in strategies
-------------------

``exhaustive`` enumerates every discrete configuration. ``random`` proposes
uniform normalized points. ``simulated_annealing`` owns an accepted state and
perturbs it with a cooling radius. ``bayesian_optimization`` owns its requested
points and queries their runtimes to fit a bounded RBF surrogate, selecting a
lower-confidence-bound proposal.

``learned_hybrid`` loads a compact offline-trained candidate ranker and scores
a bounded, deterministically replenished candidate pool in batches. It reserves
part of its recommendations for diverse or uncertain points. The shared model
remains frozen while a small residual adapter learns from retired measurements
in the current context and re-sorts only the active pool. The core tuner still
stores its full legality and runtime-history bookkeeping; the pool specifically
bounds learned inference and learned candidate metadata. Model
training and campaign data live in the separate ``alpakaTune-ml`` repository;
only a promoted deployment artifact may be bundled here.

Select one in YAML:

.. code-block:: yaml

   tuning:
     strategy: bayesian_optimization
     random_seed: 17

Custom strategies derive from ``alpakaTune::ParameterStrategy`` and implement
``recommend(StrategyContext const&)``. The only required output is a valid
normalized vector.
