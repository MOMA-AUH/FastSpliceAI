import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pysam

from spliceai import __version__
from spliceai.__main__ import add_spliceai_header, get_options, main
from spliceai.scoring import DEFAULT_BATCH_SIZE


class TestOptions(unittest.TestCase):
    def test_parses_defaults_and_core_inference_options(self):
        with (
            patch.object(
                sys,
                "argv",
                ["spliceai", "-R", "reference.fa", "-A", "grch37"],
            ),
            patch("spliceai.__main__.torch.cuda.is_available", return_value=False),
        ):
            defaults = get_options()

        self.assertEqual(defaults.output_type, "v")
        self.assertEqual(defaults.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(defaults.device, "cpu")
        self.assertFalse(defaults.mixed_precision)
        self.assertFalse(defaults.compile)

        argv = [
            "spliceai",
            "-I",
            "input.vcf",
            "-O",
            "output.vcf.gz",
            "-R",
            "reference.fa",
            "-A",
            "grch38",
            "-D",
            "100",
            "-M",
            "1",
            "--output-type",
            "z",
            "--write-index=tbi",
            "--overwrite-existing",
            "--threads",
            "2",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--mixed-precision",
            "--compile",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("spliceai.__main__.torch.set_num_threads") as set_num_threads,
            patch(
                "spliceai.__main__.torch.amp.autocast_mode.is_autocast_available",
                return_value=True,
            ),
        ):
            args = get_options()

        self.assertEqual(
            (args.I, args.O, args.R, args.A, args.D, args.M),
            ("input.vcf", "output.vcf.gz", "reference.fa", "grch38", 100, 1),
        )
        self.assertEqual((args.output_type, args.write_index), ("z", "tbi"))
        self.assertEqual((args.batch_size, args.threads, args.device), (16, 2, "cpu"))
        self.assertTrue(args.overwrite_existing)
        self.assertTrue(args.mixed_precision)
        self.assertTrue(args.compile)
        set_num_threads.assert_called_once_with(2)

    def test_existing_header_requires_explicit_overwrite(self):
        header = pysam.VariantHeader()
        header.add_line(
            '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="old version">'
        )

        with self.assertRaisesRegex(ValueError, "--overwrite-existing"):
            add_spliceai_header(header)

        add_spliceai_header(header, overwrite_existing=True)
        self.assertIn(f"SpliceAIv{__version__}", header.info["SpliceAI"].description)


class TestMain(unittest.TestCase):
    @staticmethod
    def args(**overrides):
        values = {
            "I": "input.vcf",
            "O": "output.vcf",
            "R": "reference.fa",
            "A": "grch38",
            "D": 50,
            "M": 0,
            "batch_size": 8,
            "threads": None,
            "device": "cpu",
            "mixed_precision": False,
            "compile": False,
            "output_type": "v",
            "overwrite_existing": False,
            "write_index": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def write_input_vcf(path):
        path.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=100>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t10\t.\tA\tC\t.\tPASS\t.\n"
        )

    def test_configures_inference_and_closes_resources(self):
        args = self.args(
            threads=32,
            mixed_precision=True,
            compile=True,
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
            patch("spliceai.__main__.prepare_output", return_value=(args.O, None)),
            patch(
                "spliceai.__main__.pysam.VariantFile",
                side_effect=(input_vcf, output_vcf),
            ),
            patch(
                "spliceai.__main__.EnsembleSpliceAIModel", return_value=model
            ) as model_type,
            patch("spliceai.__main__.TranscriptAnnotations", return_value=annotations),
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
        reference.close.assert_called_once_with()
        input_vcf.close.assert_called_once_with()
        output_vcf.close.assert_called_once_with()

    def test_overwrite_removes_stale_record_annotations(self):
        args = self.args(overwrite_existing=True)
        header = pysam.VariantHeader()
        header.add_line(
            '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="old version">'
        )
        record = SimpleNamespace(info={"SpliceAI": ("old",)})
        input_vcf = MagicMock(header=header)
        output_vcf = MagicMock()
        scorer = MagicMock()
        scorer.score_batch.return_value = [(record, [])]

        with (
            patch("spliceai.__main__.configure_process"),
            patch("spliceai.__main__.get_options", return_value=args),
            patch("spliceai.__main__.prepare_output", return_value=(args.O, None)),
            patch(
                "spliceai.__main__.pysam.VariantFile",
                side_effect=(input_vcf, output_vcf),
            ),
            patch("spliceai.__main__.EnsembleSpliceAIModel", return_value=MagicMock()),
            patch("spliceai.__main__.TranscriptAnnotations"),
            patch("spliceai.__main__.Fasta", return_value=MagicMock()),
            patch("spliceai.__main__.SplicingScorer", return_value=scorer),
        ):
            self.assertEqual(main(), 0)

        self.assertNotIn("SpliceAI", record.info)
        self.assertIn(f"SpliceAIv{__version__}", header.info["SpliceAI"].description)
        output_vcf.write.assert_called_once_with(record)

    def test_preserves_existing_output_when_scoring_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "input.vcf"
            output_path = directory / "output.vcf"
            self.write_input_vcf(input_path)
            output_path.write_text("existing output\n")
            args = self.args(I=input_path, O=output_path)

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
                    "spliceai.__main__.EnsembleSpliceAIModel",
                    return_value=MagicMock(),
                ),
                patch("spliceai.__main__.TranscriptAnnotations"),
                patch("spliceai.__main__.Fasta", return_value=MagicMock()),
                patch("spliceai.__main__.SplicingScorer", return_value=scorer),
            ):
                self.assertEqual(main(), 1)

            self.assertEqual(output_path.read_text(), "existing output\n")
            self.assertEqual(list(directory.glob(".output.vcf.*.tmp")), [])

    def test_writes_readable_output_formats_and_indexes(self):
        cases = (
            ("output.vcf", "v", None),
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
                args = self.args(
                    I=input_path,
                    O=output_path,
                    output_type=output_type,
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

                with pysam.VariantFile(output_path) as output_vcf:
                    if index_format is None:
                        self.assertEqual(len(list(output_vcf)), 1)
                    else:
                        index_path = Path(f"{output_path}.{index_format}")
                        self.assertTrue(index_path.is_file())
                        self.assertEqual(len(list(output_vcf.fetch("1", 0, 100))), 1)


if __name__ == "__main__":
    unittest.main()
