#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import visualize


class VisualizeTest(unittest.TestCase):
    def test_groups_strategies_and_writes_dashboard_with_overview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for strategy, runtimes in {
                "exhaustive": [4.0e-6, 2.0e-6],
                "random": [3.0e-6, 1.5e-6],
            }.items():
                directory = root / "vectorAdd" / strategy
                directory.mkdir(parents=True)
                improvements = [
                    {
                        "candidate_index": index,
                        "execution_count": (index + 1) * 10,
                        "retired_configuration_count": index + 1,
                        "runtime_seconds": runtime,
                        "elapsed_seconds": float(index + 1),
                    }
                    for index, runtime in enumerate(runtimes)
                ]
                candidate_samples = [[runtime, runtime * 1.02] for runtime in runtimes]
                candidate_configurations = [
                    {"blockRows": str(16 << index), "simdWidth": str(1 << index)}
                    for index in range(len(runtimes))
                ]
                history = {
                    "schema_version": 8,
                    "contexts": {
                        strategy: {
                            "candidate_count": 2000,
                            "metadata": {
                                "identity_entries": ["char [10]=vector/add"],
                                "kernel": "VectorAddKernel",
                                "device": "CPU",
                                "launch_specification": "FrameSpec{1,512}",
                            },
                            "best_improvements": improvements,
                            "candidate_samples": candidate_samples,
                            "candidate_estimates": runtimes,
                            "candidate_configurations": candidate_configurations,
                        }
                    },
                }
                (directory / "complete-history.json").write_text(
                    json.dumps(history), encoding="utf-8"
                )

            collected = visualize.collect(root)
            self.assertEqual(
                set(next(iter(collected["vectorAdd"].values()))),
                {"exhaustive", "random"},
            )
            launches, runtime_microseconds = visualize.improvement_series(
                next(iter(collected["vectorAdd"].values()))["exhaustive"]["best_improvements"]
            )
            self.assertEqual(launches, [10, 20])
            self.assertEqual(runtime_microseconds, [4.0, 2.0])
            candidate_indexes, candidate_runtime_microseconds = visualize.candidate_series(
                next(iter(collected["vectorAdd"].values()))["exhaustive"]
            )
            self.assertEqual(candidate_indexes, [0, 1])
            self.assertEqual(candidate_runtime_microseconds, [4.0, 2.0])
            best = visualize.best_candidates(next(iter(collected["vectorAdd"].values())))
            self.assertEqual([candidate["candidate_index"] for candidate in best], [1, 0])
            self.assertIn("blockRows=32", visualize.format_best_candidates(best))
            written = visualize.render(root)
            self.assertEqual(
                written,
                [
                    root / "plots/vectorAdd.png",
                    root / "plots/overview.png",
                    root / "plots/best_vectorAdd.png",
                    root / "plots/best_overview.png",
                    root / "plots/index.html",
                ],
            )
            self.assertTrue(all(path.is_file() for path in written))
            dashboard = written[-1].read_text(encoding="utf-8")
            self.assertIn('data-plot="vectorAdd.png"', dashboard)
            self.assertIn('data-view="candidates"', dashboard)
            self.assertIn('data-view="best"', dashboard)
            self.assertEqual(
                visualize.STRATEGY_LABELS["learned_hybrid"], "Learned hybrid"
            )

    def test_loads_mean_reported_cuda_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "vectorAdd": {
                            "reported_runtime_kind": "kernel",
                            "mean_reported_cuda_runtime_seconds": 2.5e-6,
                        },
                        "boundaryIter": {"median_duration_seconds": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                visualize.load_baseline(root),
                {"vectorAdd": {"GpuCuda": (2.5e-6, "kernel")}},
            )

    def test_loads_executor_specific_cpu_and_gpu_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "vectorAdd": {
                            "reported_runtime_kind": "kernel",
                            "reported_runtimes": {
                                "CpuOmpBlocks": {"mean_runtime_seconds": 4.0e-6},
                                "GpuCuda": {"mean_runtime_seconds": 2.0e-6},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            baselines = visualize.load_baseline(root)

        self.assertEqual(
            baselines,
            {
                "vectorAdd": {
                    "CpuOmpBlocks": (4.0e-6, "kernel"),
                    "GpuCuda": (2.0e-6, "kernel"),
                }
            },
        )

    def test_context_executor_uses_persisted_executor_identity(self) -> None:
        key = (
            ("alpaka::exec::CpuOmpBlocks=CpuOmpBlocks",),
            "Kernel",
            "CPU",
            "FrameSpec{executor=CpuOmpBlocks}",
            10,
        )
        self.assertEqual(visualize.context_executor(key), "CpuOmpBlocks")


if __name__ == "__main__":
    unittest.main()
