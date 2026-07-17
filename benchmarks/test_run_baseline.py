#!/usr/bin/env python3

import contextlib
import io
import unittest

import run_baseline


class ParseArgumentsTest(unittest.TestCase):
    def test_selects_examples_and_repetitions(self) -> None:
        arguments = run_baseline.parse_args(
            ["--examples", "grayScale", "vectorAdd", "--repetitions", "3"]
        )
        self.assertEqual(arguments.examples, ["grayScale", "vectorAdd"])
        self.assertEqual(arguments.repetitions, 3)

    def test_rejects_non_positive_repetitions(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            run_baseline.parse_args(["--repetitions", "0"])
        self.assertIn("--repetitions must be greater than zero", stderr.getvalue())

    def test_extracts_cuda_kernel_runtime_not_host_runtime(self) -> None:
        output = """Using alpaka accelerator: cpuSerial for HOST
Average time for kernel execution: 0.5s
Using alpaka accelerator: gpuCuda for CUDA nvidiaGpu
Average time for kernel execution: 2.5e-05s
"""
        self.assertEqual(
            run_baseline.reported_cuda_runtime("vectorAdd", output),
            2.5e-05,
        )

    def test_converts_cuda_time_step_milliseconds_to_seconds(self) -> None:
        output = """HOST
Time per time step: 10 ms.
CUDA
Time per time step: 0.125 ms.
"""
        self.assertEqual(
            run_baseline.reported_cuda_runtime("heatEquation2D", output),
            0.000125,
        )


if __name__ == "__main__":
    unittest.main()
