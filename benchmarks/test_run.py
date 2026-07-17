#!/usr/bin/env python3

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import run


class ParseArgumentsTest(unittest.TestCase):
    def test_examples_selects_only_requested_benchmarks(self) -> None:
        arguments = run.parse_args(["--examples", "scan", "vectorAdd"])
        self.assertEqual(arguments.examples, ("scan", "vectorAdd"))

    def test_examples_normalizes_aliases_and_duplicates(self) -> None:
        arguments = run.parse_args(
            ["--examples", "greyScale", "matrixMultiplication", "matrixMultiplication"]
        )
        self.assertEqual(arguments.examples, ("grayScale", "matrixMultiplication"))

    def test_exclude_examples_filters_the_default_selection(self) -> None:
        arguments = run.parse_args(["--exclude-examples", "nBody", "grayScale"])
        self.assertNotIn("nBody", arguments.examples)
        self.assertNotIn("grayScale", arguments.examples)
        self.assertIn("vectorAdd", arguments.examples)

    def test_exclusions_are_applied_after_inclusions(self) -> None:
        arguments = run.parse_args(
            [
                "--examples",
                "scan",
                "vectorAdd",
                "--exclude-examples",
                "scan",
            ]
        )
        self.assertEqual(arguments.examples, ("vectorAdd",))

    def test_empty_selection_is_rejected(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            run.parse_args(
                ["--examples", "scan", "--exclude-examples", "scan"]
            )
        self.assertIn("the example selection is empty", stderr.getvalue())

    def test_default_run_keeps_safety_limits_and_all_strategies(self) -> None:
        arguments = run.parse_args(["--examples", "vectorAdd"])
        self.assertEqual(arguments.strategies, run.STRATEGIES)
        self.assertEqual(
            arguments.maximum_executions, run.DEFAULT_MAXIMUM_EXECUTIONS
        )
        self.assertEqual(
            arguments.maximum_retired_configurations,
            run.DEFAULT_MAXIMUM_RETIRED_CONFIGURATIONS,
        )

    def test_full_coverage_is_uncapped_exhaustive(self) -> None:
        arguments = run.parse_args(
            ["--examples", "heatEquation2D", "nBody", "--full-coverage"]
        )
        self.assertEqual(arguments.strategies, ("exhaustive",))
        self.assertIsNone(arguments.maximum_executions)
        self.assertIsNone(arguments.maximum_retired_configurations)

    def test_full_coverage_rejects_search_strategies(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            run.parse_args(
                [
                    "--examples",
                    "vectorAdd",
                    "--full-coverage",
                    "--strategies",
                    "random",
                ]
            )
        self.assertIn("only supports --strategies exhaustive", stderr.getvalue())

    def test_full_coverage_rejects_completion_limits(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            run.parse_args(
                [
                    "--examples",
                    "vectorAdd",
                    "--full-coverage",
                    "--maximum-executions",
                    "4000",
                ]
            )
        self.assertIn("cannot be combined with completion limits", stderr.getvalue())


class FullCoverageConfigurationTest(unittest.TestCase):
    def test_full_coverage_uses_fixed_measurements_and_omits_limits(self) -> None:
        base = {
            "schema_version": 1,
            "tuning": {
                "strategy": "random",
                "warmup_runs": 9,
                "runs_per_candidate": 20,
                "minimum_runs_per_candidate": 1,
                "mann_whitney_early_stop": True,
                "max_consecutive_runs": 10,
                "maximum_executions": 4000,
                "maximum_retired_configurations": 2000,
            },
            "persistence": {"file": "old.json"},
        }
        generated = run.benchmark_configuration(
            base,
            "exhaustive",
            Path("history.json"),
            None,
            None,
            True,
        )

        self.assertEqual(generated["tuning"]["strategy"], "exhaustive")
        for key, value in run.FULL_COVERAGE_TUNING.items():
            self.assertEqual(generated["tuning"][key], value)
        self.assertNotIn("maximum_executions", generated["tuning"])
        self.assertNotIn("maximum_retired_configurations", generated["tuning"])
        self.assertEqual(generated["persistence"]["file"], "history.json")
        self.assertIn("maximum_executions", base["tuning"])

    def test_resume_requires_a_verified_full_coverage_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "history.json").write_text("{}", encoding="utf-8")
            (directory / "run.json").write_text(
                json.dumps({"status": "completed", "full_coverage_verified": False}),
                encoding="utf-8",
            )
            self.assertTrue(run.successful_run(directory))
            self.assertFalse(run.successful_run(directory, require_full_coverage=True))


class HistoryInspectionTest(unittest.TestCase):
    def inspect(self, context: dict) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary) / "history.json"
            history.write_text(
                json.dumps({"schema_version": 9, "contexts": {"abc": context}}),
                encoding="utf-8",
            )
            return run.inspect_history(history)

    @staticmethod
    def context(reason: str, retired: int, rejected: list[bool]) -> dict:
        return {
            "candidate_count": len(rejected),
            "retired_configuration_count": retired,
            "rejected_candidates": rejected,
            "completion_reason": reason,
            "metadata": {"kernel": "ExampleKernel"},
        }

    def test_complete_history_accounts_for_measured_and_rejected_candidates(self) -> None:
        diagnostics = self.inspect(
            self.context("all_configurations", 3, [False, True, False, False])
        )
        self.assertTrue(diagnostics["valid"])
        self.assertTrue(diagnostics["all_contexts_complete"])
        self.assertEqual(diagnostics["messages"], [])
        self.assertEqual(diagnostics["contexts"][0]["legal_candidate_count"], 3)
        self.assertEqual(diagnostics["contexts"][0]["coverage"], 1.0)

    def test_execution_cap_explains_partial_exhaustive_coverage(self) -> None:
        diagnostics = self.inspect(
            self.context("maximum_executions", 2, [False, False, False, False])
        )
        self.assertFalse(diagnostics["all_contexts_complete"])
        message = diagnostics["messages"][0]
        self.assertIn("2/4 legal candidates (50.0%)", message)
        self.assertIn("warm-ups and repeated measurements", message)
        self.assertIn("does not indicate skipped exhaustive candidates", message)

    def test_process_exit_before_completion_is_not_full_coverage(self) -> None:
        diagnostics = self.inspect(
            self.context("none", 3, [False, False, False, False])
        )
        self.assertFalse(diagnostics["all_contexts_complete"])
        self.assertIn("exited before tuning completed", diagnostics["messages"][0])


if __name__ == "__main__":
    unittest.main()
