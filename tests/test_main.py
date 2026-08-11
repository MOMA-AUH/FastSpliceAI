import argparse
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pysam

from spliceai import __version__
from spliceai.__main__ import (
    add_spliceai_header,
    configure_process,
    get_options,
    log_scoring_progress,
    main,
    paths_refer_to_same_file,
    positive_int,
)
from spliceai.scoring import DEFAULT_BATCH_SIZE


class TestOptions(unittest.TestCase):
    def test_configures_process_only_when_cli_runs(self):
        with (
            patch("spliceai.__main__.logging.basicConfig") as configure_logging,
            patch("spliceai.__main__.signal.signal") as configure_signal,
        ):
            configure_process()

        configure_logging.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        configure_signal.assert_called_once()

    def test_batch_size_defaults_to_eight(self):
        argv = ["spliceai", "-R", "reference.fa", "-A", "grch37"]

        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
        ):
            args = get_options()

        self.assertEqual(args.batch_size, DEFAULT_BATCH_SIZE)
        self.assertIsNone(args.threads)
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.overwrite_existing)

    def test_accepts_explicit_batch_size(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--batch-size",
            "32",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
        ):
            args = get_options()

        self.assertEqual(args.batch_size, 32)

    def test_rejects_non_positive_batch_size(self):
        for value in ("0", "-1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    argparse.ArgumentTypeError, "must be a positive integer"
                ),
            ):
                positive_int(value)

    def test_accepts_explicit_thread_count(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--threads",
            "32",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.set_num_threads") as set_num_threads,
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
        ):
            args = get_options()

        self.assertEqual(args.threads, 32)
        set_num_threads.assert_called_once_with(32)

    def test_accepts_explicit_cpu_device(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--device",
            "cpu",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available") as cuda_available,
        ):
            args = get_options()

        self.assertEqual(args.device, "cpu")
        cuda_available.assert_not_called()

    def test_unavailable_explicit_cuda_falls_back_to_cpu(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--device",
            "cuda",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
            patch("spliceai.__main__.logger.warning") as warning,
        ):
            args = get_options()

        self.assertEqual(args.device, "cpu")
        warning.assert_called_once_with("CUDA is not available, falling back to CPU")

    def test_auto_device_uses_cuda_when_available(self):
        argv = ["spliceai", "-R", "reference.fa", "-A", "grch37"]

        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=True),
        ):
            args = get_options()

        self.assertEqual(args.device, "cuda")

    def test_detects_equivalent_input_and_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.vcf"
            input_path.touch()
            hard_link = Path(directory) / "hard-link.vcf"
            os.link(input_path, hard_link)

            self.assertTrue(paths_refer_to_same_file(input_path, input_path.resolve()))
            self.assertTrue(paths_refer_to_same_file(input_path, hard_link))
            self.assertFalse(
                paths_refer_to_same_file(input_path, Path(directory) / "output.vcf")
            )

    def test_existing_header_requires_explicit_overwrite(self):
        header = pysam.VariantHeader()
        header.add_line(
            '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="old version">'
        )

        with self.assertRaisesRegex(ValueError, "--overwrite-existing"):
            add_spliceai_header(header)

        add_spliceai_header(header, overwrite_existing=True)
        self.assertIn(f"SpliceAIv{__version__}", header.info["SpliceAI"].description)


class TestProgressLogging(unittest.TestCase):
    def test_logs_progress_at_thirty_second_intervals(self):
        records = [
            ("record-1", []),
            ("record-2", []),
            ("record-3", []),
            ("record-4", []),
        ]

        with (
            patch(
                "spliceai.__main__.time.monotonic",
                side_effect=(0.0, 29.0, 30.0, 59.0, 60.0, 60.0),
            ),
            patch("spliceai.__main__.logger.info") as info,
        ):
            self.assertEqual(list(log_scoring_progress(records)), records)

        self.assertEqual(
            info.call_args_list,
            [
                call(
                    "Scored %d records in %.1f seconds (%.1f records/second)",
                    2,
                    30.0,
                    2 / 30,
                ),
                call(
                    "Scored %d records in %.1f seconds (%.1f records/second)",
                    4,
                    60.0,
                    4 / 60,
                ),
                call(
                    "Finished scoring %d records in %.1f seconds (%.1f records/second)",
                    4,
                    60.0,
                    4 / 60,
                ),
            ],
        )

    def test_logs_completion_for_short_and_empty_streams(self):
        cases = (([], 0, 0.0), (["record"], 1, 0.5))

        for records, expected_count, expected_rate in cases:
            with (
                self.subTest(records=records),
                patch(
                    "spliceai.__main__.time.monotonic",
                    side_effect=(10.0, 12.0, 12.0) if records else (10.0, 12.0),
                ),
                patch("spliceai.__main__.logger.info") as info,
            ):
                self.assertEqual(list(log_scoring_progress(records)), records)

            self.assertEqual(
                info.call_args_list,
                [
                    call(
                        "Finished scoring %d records in %.1f seconds "
                        "(%.1f records/second)",
                        expected_count,
                        2.0,
                        expected_rate,
                    ),
                ],
            )


class TestMain(unittest.TestCase):
    def test_closes_resources_after_success(self):
        args = SimpleNamespace(
            I="input.vcf",
            O="output.vcf",
            R="reference.fa",
            A="grch38",
            D=50,
            M=0,
            batch_size=8,
            threads=32,
            device="cpu",
            overwrite_existing=False,
        )
        input_vcf = MagicMock()
        input_vcf.header = MagicMock()
        output_vcf = MagicMock()
        model = MagicMock()
        annotations = MagicMock()
        reference = MagicMock()
        scorer = MagicMock()
        scorer.score_batch.return_value = []

        with (
            patch("spliceai.__main__.configure_process"),
            patch("spliceai.__main__.get_options", return_value=args),
            patch(
                "spliceai.__main__.pysam.VariantFile",
                side_effect=(input_vcf, output_vcf),
            ),
            patch(
                "spliceai.__main__.EnsembleSpliceAIModel", return_value=model
            ) as model_type,
            patch(
                "spliceai.__main__.TranscriptAnnotations",
                return_value=annotations,
            ),
            patch("spliceai.__main__.Fasta", return_value=reference),
            patch(
                "spliceai.__main__.SplicingScorer", return_value=scorer
            ) as scorer_type,
        ):
            self.assertEqual(main(), 0)

        model_type.assert_called_once_with()
        model.to.assert_called_once_with("cpu")
        scorer_type.assert_called_once_with(
            model=model.to.return_value,
            transcript_annotations=annotations,
            ref_fasta=reference,
            distance=50,
            mask=0,
            batch_size=8,
        )
        reference.close.assert_called_once_with()
        input_vcf.close.assert_called_once_with()
        output_vcf.close.assert_called_once_with()

    def test_does_not_open_output_when_model_setup_fails(self):
        args = SimpleNamespace(
            I="input.vcf",
            O="output.vcf",
            R="reference.fa",
            A="grch38",
            D=50,
            M=0,
            batch_size=8,
            threads=None,
            device="cuda",
            overwrite_existing=False,
        )
        input_vcf = MagicMock()

        with (
            patch("spliceai.__main__.configure_process"),
            patch("spliceai.__main__.get_options", return_value=args),
            patch("spliceai.__main__.add_spliceai_header"),
            patch(
                "spliceai.__main__.pysam.VariantFile", return_value=input_vcf
            ) as variant_file,
            patch(
                "spliceai.__main__.EnsembleSpliceAIModel",
                side_effect=ValueError("CUDA unavailable"),
            ),
        ):
            self.assertEqual(main(), 1)

        variant_file.assert_called_once_with("input.vcf")
        input_vcf.close.assert_called_once_with()

    def test_rejects_same_input_and_output_before_opening(self):
        args = SimpleNamespace(
            I="variants.vcf",
            O="variants.vcf",
            R="reference.fa",
            A="grch38",
            D=50,
            M=0,
            batch_size=8,
            threads=None,
            device="cpu",
            overwrite_existing=False,
        )
        with (
            patch("spliceai.__main__.configure_process"),
            patch("spliceai.__main__.get_options", return_value=args),
            patch("spliceai.__main__.pysam.VariantFile") as variant_file,
        ):
            self.assertEqual(main(), 2)
        variant_file.assert_not_called()

    def test_overwrite_removes_stale_record_annotations(self):
        args = SimpleNamespace(
            I="input.vcf",
            O="output.vcf",
            R="reference.fa",
            A="grch38",
            D=50,
            M=0,
            batch_size=8,
            threads=None,
            device="auto",
            overwrite_existing=True,
        )
        header = pysam.VariantHeader()
        header.add_line(
            '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="old version">'
        )
        record = SimpleNamespace(info={"SpliceAI": ("old",)})
        input_vcf = MagicMock(header=header)
        output_vcf = MagicMock()
        model = MagicMock()
        annotations = MagicMock()
        reference = MagicMock()
        scorer = MagicMock()
        scorer.score_batch.return_value = [(record, [])]

        with (
            patch("spliceai.__main__.configure_process"),
            patch("spliceai.__main__.get_options", return_value=args),
            patch(
                "spliceai.__main__.pysam.VariantFile",
                side_effect=(input_vcf, output_vcf),
            ),
            patch("spliceai.__main__.EnsembleSpliceAIModel", return_value=model),
            patch(
                "spliceai.__main__.TranscriptAnnotations",
                return_value=annotations,
            ),
            patch("spliceai.__main__.Fasta", return_value=reference),
            patch("spliceai.__main__.SplicingScorer", return_value=scorer),
        ):
            self.assertEqual(main(), 0)

        self.assertNotIn("SpliceAI", record.info)
        self.assertIn(f"SpliceAIv{__version__}", header.info["SpliceAI"].description)
        output_vcf.write.assert_called_once_with(record)
        reference.close.assert_called_once_with()
