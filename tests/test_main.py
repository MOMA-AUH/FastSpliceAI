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
    validate_index_options,
    variant_output_mode,
)
from spliceai.scoring import DEFAULT_BATCH_SIZE


class TestOptions(unittest.TestCase):
    def test_output_type_defaults_to_uncompressed_vcf(self):
        argv = ["spliceai", "-R", "reference.fa", "-A", "grch37"]
        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
        ):
            args = get_options()

        self.assertEqual(args.output_type, "v")

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
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--output-type",
            "v",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
        ):
            args = get_options()

        self.assertEqual(args.batch_size, DEFAULT_BATCH_SIZE)
        self.assertIsNone(args.threads)
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.mixed_precision)
        self.assertFalse(args.allow_fallback)
        self.assertFalse(args.compile)
        self.assertFalse(args.overwrite_existing)

    def test_accepts_explicit_batch_size(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--output-type",
            "v",
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
            "--output-type",
            "v",
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
            "--output-type",
            "v",
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

    def test_accepts_available_cuda_device(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--output-type",
            "v",
            "--device",
            "cuda",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=True),
        ):
            args = get_options()

        self.assertEqual(args.device, "cuda")

    def test_rejects_unavailable_cuda_device(self):
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
            self.assertRaisesRegex(RuntimeError, "CUDA was requested"),
        ):
            get_options()

    def test_falls_back_from_unavailable_cuda_when_allowed(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--device",
            "cuda",
            "--allow-fallback",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
            patch("spliceai.__main__.logger.warning") as warning,
        ):
            args = get_options()

        self.assertEqual(args.device, "cpu")
        warning.assert_called_once_with(
            "CUDA was requested but is not available; falling back to CPU"
        )

    def test_resolves_auto_device(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
        ]

        for cuda_available, expected_device in ((False, "cpu"), (True, "cuda")):
            with (
                self.subTest(cuda_available=cuda_available),
                patch.object(sys, "argv", argv),
                patch(
                    "spliceai.__main__.torch.cuda.is_available",
                    return_value=cuda_available,
                ),
            ):
                args = get_options()

            self.assertEqual(args.device, expected_device)

    def test_accepts_supported_mixed_precision(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--device",
            "cpu",
            "--mixed-precision",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "spliceai.__main__.torch.amp.autocast_mode.is_autocast_available",
                return_value=True,
            ) as autocast_available,
        ):
            args = get_options()

        self.assertTrue(args.mixed_precision)
        autocast_available.assert_called_once_with("cpu")

    def test_accepts_model_compilation(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--compile",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
        ):
            args = get_options()

        self.assertTrue(args.compile)

    def test_rejects_unavailable_mixed_precision(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--device",
            "cpu",
            "--mixed-precision",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "spliceai.__main__.torch.amp.autocast_mode.is_autocast_available",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "mixed precision was requested"),
        ):
            get_options()

    def test_falls_back_from_unavailable_mixed_precision_when_allowed(self):
        argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--device",
            "cpu",
            "--mixed-precision",
            "--allow-fallback",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "spliceai.__main__.torch.amp.autocast_mode.is_autocast_available",
                return_value=False,
            ),
            patch("spliceai.__main__.logger.warning") as warning,
        ):
            args = get_options()

        self.assertFalse(args.mixed_precision)
        warning.assert_called_once_with(
            "mixed precision was requested but is not available; falling back to float32"
        )

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

    def test_selects_output_mode_from_explicit_type(self):
        self.assertEqual(variant_output_mode("v"), "w")
        self.assertEqual(variant_output_mode("z"), "wz")
        self.assertEqual(variant_output_mode("b"), "wb")

    def test_write_index_defaults_to_csi_and_accepts_tbi(self):
        base_argv = [
            "spliceai",
            "-R",
            "reference.fa",
            "-A",
            "grch37",
            "--output-type",
            "z",
        ]
        with patch.object(sys, "argv", [*base_argv, "--write-index"]):
            self.assertEqual(get_options().write_index, "csi")
        with patch.object(sys, "argv", [*base_argv, "--write-index=tbi"]):
            self.assertEqual(get_options().write_index, "tbi")

    def test_validates_index_compatibility(self):
        validate_index_options("output.vcf.gz", "z", "csi")
        validate_index_options("output.vcf.gz", "z", "tbi")
        validate_index_options("output.bcf", "b", "csi")
        cases = (
            (sys.stdout, "z", "csi", "filesystem output path"),
            ("-", "z", "csi", "standard output"),
            ("output.vcf", "v", "csi", "compressed"),
            ("output.bcf", "b", "tbi", "only supported"),
        )
        for output, output_type, index_format, message in cases:
            with self.subTest(output=output, output_type=output_type):
                with self.assertRaisesRegex(ValueError, message):
                    validate_index_options(output, output_type, index_format)


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
    @staticmethod
    def write_input_vcf(path):
        path.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=100>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t10\t.\tA\tC\t.\tPASS\t.\n"
        )

    def test_configures_inference_and_closes_resources_after_success(self):
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
            mixed_precision=True,
            compile=True,
            output_type="v",
            overwrite_existing=False,
            write_index=None,
        )
        input_vcf = MagicMock()
        input_vcf.header = MagicMock()
        output_vcf = MagicMock()
        model = MagicMock()
        device_model = model.to.return_value
        original_forward = device_model.forward
        compiled_forward = MagicMock()
        annotations = MagicMock()
        reference = MagicMock()
        scorer = MagicMock()
        scorer.score_batch.return_value = []

        with (
            patch("spliceai.__main__.configure_process"),
            patch("spliceai.__main__.get_options", return_value=args),
            patch(
                "spliceai.__main__.prepare_output",
                return_value=(args.O, None),
            ),
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
            patch(
                "spliceai.__main__.torch.compile", return_value=compiled_forward
            ) as compile_model,
            patch("spliceai.__main__.torch.inference_mode") as inference_mode,
            patch("spliceai.__main__.torch.autocast") as autocast,
        ):
            self.assertEqual(main(), 0)

        model_type.assert_called_once_with()
        model.to.assert_called_once_with("cpu")
        compile_model.assert_called_once_with(
            original_forward, dynamic=True, fullgraph=True
        )
        self.assertIs(device_model.forward, compiled_forward)
        scorer_type.assert_called_once_with(
            model=device_model,
            transcript_annotations=annotations,
            ref_fasta=reference,
            distance=50,
            mask=0,
            batch_size=8,
        )
        inference_mode.assert_called_once_with()
        autocast.assert_called_once_with(device_type="cpu", enabled=True)
        inference_mode.return_value.__enter__.assert_called_once_with()
        autocast.return_value.__enter__.assert_called_once_with()
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
            mixed_precision=False,
            compile=False,
            output_type="v",
            overwrite_existing=False,
            write_index=None,
        )
        input_vcf = MagicMock()
        model = MagicMock()
        model.to.side_effect = RuntimeError("CUDA initialization failed")

        with (
            patch("spliceai.__main__.configure_process"),
            patch("spliceai.__main__.get_options", return_value=args),
            patch("spliceai.__main__.add_spliceai_header"),
            patch(
                "spliceai.__main__.pysam.VariantFile", return_value=input_vcf
            ) as variant_file,
            patch(
                "spliceai.__main__.EnsembleSpliceAIModel",
                return_value=model,
            ),
        ):
            self.assertEqual(main(), 1)

        variant_file.assert_called_once_with("input.vcf")
        model.to.assert_called_once_with("cuda")
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
            mixed_precision=False,
            compile=False,
            output_type="v",
            overwrite_existing=False,
            write_index=None,
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
            device="cpu",
            mixed_precision=False,
            compile=False,
            output_type="v",
            overwrite_existing=True,
            write_index=None,
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
                "spliceai.__main__.prepare_output",
                return_value=(args.O, None),
            ),
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

    def test_preserves_existing_output_when_scoring_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "input.vcf"
            output_path = directory / "output.vcf"
            self.write_input_vcf(input_path)
            output_path.write_text("existing output\n")
            args = SimpleNamespace(
                I=input_path,
                O=output_path,
                R="reference.fa",
                A="grch38",
                D=50,
                M=0,
                batch_size=8,
                threads=None,
                device="cpu",
                mixed_precision=False,
                compile=False,
                output_type="v",
                overwrite_existing=False,
                write_index=None,
            )

            def fail_after_first_record(records):
                for record in records:
                    yield record, []
                    raise ValueError("scoring failed")

            scorer = MagicMock()
            scorer.score_batch.side_effect = fail_after_first_record
            with (
                patch("spliceai.__main__.configure_process"),
                patch("spliceai.__main__.get_options", return_value=args),
                patch(
                    "spliceai.__main__.EnsembleSpliceAIModel", return_value=MagicMock()
                ),
                patch("spliceai.__main__.TranscriptAnnotations"),
                patch("spliceai.__main__.Fasta", return_value=MagicMock()),
                patch("spliceai.__main__.SplicingScorer", return_value=scorer),
            ):
                self.assertEqual(main(), 1)

            self.assertEqual(output_path.read_text(), "existing output\n")
            self.assertEqual(list(directory.glob(".output.vcf.*.tmp")), [])

    def test_preserves_existing_output_when_indexing_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "input.vcf"
            output_path = directory / "output.vcf.gz"
            index_path = Path(f"{output_path}.csi")
            self.write_input_vcf(input_path)
            output_path.write_text("existing output\n")
            index_path.write_text("existing index\n")
            args = SimpleNamespace(
                I=input_path,
                O=output_path,
                R="reference.fa",
                A="grch38",
                D=50,
                M=0,
                batch_size=8,
                threads=None,
                device="cpu",
                mixed_precision=False,
                compile=False,
                output_type="z",
                overwrite_existing=False,
                write_index="csi",
            )
            scorer = MagicMock()
            scorer.score_batch.side_effect = lambda records: (
                (record, []) for record in records
            )
            with (
                patch("spliceai.__main__.configure_process"),
                patch("spliceai.__main__.get_options", return_value=args),
                patch(
                    "spliceai.__main__.EnsembleSpliceAIModel", return_value=MagicMock()
                ),
                patch("spliceai.__main__.TranscriptAnnotations"),
                patch("spliceai.__main__.Fasta", return_value=MagicMock()),
                patch("spliceai.__main__.SplicingScorer", return_value=scorer),
                patch(
                    "spliceai.__main__.prepare_index",
                    side_effect=pysam.SamtoolsError("indexing failed"),
                ),
            ):
                self.assertEqual(main(), 1)

            self.assertEqual(output_path.read_text(), "existing output\n")
            self.assertEqual(index_path.read_text(), "existing index\n")
            self.assertEqual(list(directory.glob(".*.tmp")), [])

    def test_writes_readable_output_formats_atomically(self):
        cases = (
            (".vcf", "v", b"##"),
            (".vcf.gz", "z", b"\x1f\x8b"),
            (".bcf", "b", b"\x1f\x8b"),
            (".bcf.gz", "b", b"\x1f\x8b"),
            (".bcf.bgz", "b", b"\x1f\x8b"),
        )
        for suffix, output_type, expected_prefix in cases:
            with (
                self.subTest(suffix=suffix),
                tempfile.TemporaryDirectory() as directory,
            ):
                directory = Path(directory)
                input_path = directory / "input.vcf"
                output_path = directory / f"output{suffix}"
                self.write_input_vcf(input_path)
                args = SimpleNamespace(
                    I=input_path,
                    O=output_path,
                    R="reference.fa",
                    A="grch38",
                    D=50,
                    M=0,
                    batch_size=8,
                    threads=None,
                    device="cpu",
                    mixed_precision=False,
                    compile=False,
                    output_type=output_type,
                    overwrite_existing=False,
                    write_index=None,
                )
                scorer = MagicMock()
                scorer.score_batch.side_effect = lambda records: (
                    (record, []) for record in records
                )
                with (
                    patch("spliceai.__main__.configure_process"),
                    patch("spliceai.__main__.get_options", return_value=args),
                    patch(
                        "spliceai.__main__.EnsembleSpliceAIModel",
                        return_value=MagicMock(),
                    ),
                    patch("spliceai.__main__.TranscriptAnnotations"),
                    patch("spliceai.__main__.Fasta", return_value=MagicMock()),
                    patch("spliceai.__main__.SplicingScorer", return_value=scorer),
                ):
                    self.assertEqual(main(), 0)

                self.assertEqual(output_path.read_bytes()[:2], expected_prefix)
                with pysam.VariantFile(output_path) as output_vcf:
                    self.assertEqual(len(list(output_vcf)), 1)
                self.assertEqual(list(directory.glob(f".{output_path.name}.*.tmp")), [])

    def test_writes_requested_indexes(self):
        cases = (
            ("output.vcf.gz", "z", "csi"),
            ("output.vcf.gz", "z", "tbi"),
            ("output.bcf", "b", "csi"),
        )
        for output_name, output_type, index_format in cases:
            with (
                self.subTest(output_type=output_type, index_format=index_format),
                tempfile.TemporaryDirectory() as directory,
            ):
                directory = Path(directory)
                input_path = directory / "input.vcf"
                output_path = directory / output_name
                self.write_input_vcf(input_path)
                args = SimpleNamespace(
                    I=input_path,
                    O=output_path,
                    R="reference.fa",
                    A="grch38",
                    D=50,
                    M=0,
                    batch_size=8,
                    threads=None,
                    device="cpu",
                    mixed_precision=False,
                    compile=False,
                    output_type=output_type,
                    overwrite_existing=False,
                    write_index=index_format,
                )
                scorer = MagicMock()
                scorer.score_batch.side_effect = lambda records: (
                    (record, []) for record in records
                )
                with (
                    patch("spliceai.__main__.configure_process"),
                    patch("spliceai.__main__.get_options", return_value=args),
                    patch(
                        "spliceai.__main__.EnsembleSpliceAIModel",
                        return_value=MagicMock(),
                    ),
                    patch("spliceai.__main__.TranscriptAnnotations"),
                    patch("spliceai.__main__.Fasta", return_value=MagicMock()),
                    patch("spliceai.__main__.SplicingScorer", return_value=scorer),
                ):
                    self.assertEqual(main(), 0)

                index_path = Path(f"{output_path}.{index_format}")
                self.assertTrue(index_path.is_file())
                with pysam.VariantFile(output_path) as output_vcf:
                    self.assertEqual(len(list(output_vcf.fetch("1", 0, 100))), 1)
