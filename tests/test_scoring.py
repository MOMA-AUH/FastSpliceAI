import unittest
from collections import namedtuple
from importlib.resources import files
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pyfaidx import Fasta

from spliceai.annotation import TranscriptAnnotations
from spliceai.model import EnsembleSpliceAIModel
from spliceai.scoring import DEFAULT_BATCH_SIZE, SplicingScorer
from spliceai.utils import one_hot_encode
from tests import name

torch.set_num_threads(2)

Record = namedtuple("Record", ["chrom", "pos", "ref", "alts"])


class Sequence:
    def __init__(self, sequence):
        self.sequence = sequence

    def __getitem__(self, key):
        return SimpleNamespace(seq=self.sequence[key])


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.batch_shapes = []

    def infer(self, inputs):
        self.calls += 1
        self.batch_shapes.append(inputs.shape)
        output_length = inputs.shape[1] - 10000
        return np.zeros((inputs.shape[0], output_length, 3), dtype=np.float32)


class MaskingModel(CountingModel):
    def infer(self, inputs):
        predictions = super().infer(inputs)
        predictions[1::2, 2, 1] = 0.8
        predictions[1::2, 3, 2] = 0.7
        return predictions


class CapturingModel(CountingModel):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def infer(self, inputs):
        self.inputs.append(inputs.copy())
        return super().infer(inputs)


class StubAnnotations:
    def __init__(self):
        self.genes = [
            SimpleNamespace(strand="+", value={"name": "GENE_A", "order": 0}),
            SimpleNamespace(strand="-", value={"name": "GENE_B", "order": 1}),
        ]
        self.ref_fasta = {"1": Sequence("A" * 12000)}
        self.model = CountingModel()
        self.position_data_calls = 0

    def get_overlapping_genes(self, chrom, pos):
        return self.genes

    def get_pos_data(self, gene, pos):
        self.position_data_calls += 1
        return 0, 0, 0


def make_scorer(
    annotator,
    distance,
    mask,
    batch_size=DEFAULT_BATCH_SIZE,
    model=None,
):
    return SplicingScorer(
        model=annotator.model if model is None else model,
        transcript_annotations=annotator,
        ref_fasta=annotator.ref_fasta,
        distance=distance,
        mask=mask,
        batch_size=batch_size,
    )


def score_record(record, annotator, distance, mask):
    return make_scorer(annotator, distance, mask).score(record)


def score_records(records, annotator, distance, mask, batch_size):
    return make_scorer(annotator, distance, mask, batch_size).score_batch(records)


class TestDeltaScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fasta_path = files(name).joinpath("data/test.fa")
        fasta_without_prefix_path = files(name).joinpath("data/test_without_prefix.fa")
        model = EnsembleSpliceAIModel()
        cls.ann = TranscriptAnnotations("grch37")
        cls.ann.model = model
        cls.ann.ref_fasta = Fasta(fasta_path, rebuild=False)
        cls.ann_without_prefix = TranscriptAnnotations("grch37")
        cls.ann_without_prefix.model = model
        cls.ann_without_prefix.ref_fasta = Fasta(
            fasta_without_prefix_path, rebuild=False
        )

    @classmethod
    def tearDownClass(cls):
        cls.ann.ref_fasta.close()
        cls.ann_without_prefix.ref_fasta.close()

    def test_get_delta_score_acceptor(self):
        record = Record("10", 94077, "A", ["C"])
        scores = score_record(record, self.ann, 500, 0)
        self.assertEqual(scores, ["C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193"])
        scores = score_record(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ["C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193"])

        record = Record("chr10", 94077, "A", ["C"])
        scores = score_record(record, self.ann, 500, 0)
        self.assertEqual(scores, ["C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193"])
        scores = score_record(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ["C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193"])

    def test_get_delta_score_donor(self):
        record = Record("10", 94555, "C", ["T"])
        scores = score_record(record, self.ann, 500, 0)
        self.assertEqual(scores, ["T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0"])
        scores = score_record(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ["T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0"])

        record = Record("chr10", 94555, "C", ["T"])
        scores = score_record(record, self.ann, 500, 0)
        self.assertEqual(scores, ["T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0"])
        scores = score_record(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ["T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0"])

    def test_get_delta_score_indels_and_masking(self):
        insertion = Record("10", 94077, "A", ["AC"])
        deletion = Record("10", 94077, "AC", ["A"])

        raw_scores = list(score_records([insertion, deletion], self.ann, 500, 0, 4))
        masked_scores = list(score_records([insertion, deletion], self.ann, 500, 1, 4))

        self.assertEqual(
            [scores for _, scores in raw_scores],
            [
                ["AC|TUBB8|0.11|0.02|0.00|0.02|89|-23|1|-22"],
                ["A|TUBB8|0.02|0.03|0.03|0.04|145|0|174|-22"],
            ],
        )
        self.assertEqual(
            [scores for _, scores in masked_scores],
            [
                ["AC|TUBB8|0.11|0.02|0.00|0.00|89|-23|1|-22"],
                ["A|TUBB8|0.02|0.00|0.03|0.00|145|0|174|-22"],
            ],
        )


class TestDeltaScoreOptimizations(unittest.TestCase):
    def test_preparation_defers_encoding_until_batch_assembly(self):
        ann = StubAnnotations()
        record = Record("1", 6000, "A", ["C", "G", "T"])

        with patch("spliceai.scoring.one_hot_encode_into") as encode:
            encode.side_effect = lambda sequence, destination, **kwargs: (
                destination.fill(0)
            )
            scorer = make_scorer(ann, 2, 0)
            prepared = scorer._prepare_record(record)
            self.assertEqual(encode.call_count, 0)
            tasks = list(prepared.tasks)
            self.assertEqual(encode.call_count, 0)

            scorer._infer_batch(tasks)

            self.assertEqual(encode.call_count, len(tasks))

    def test_uses_an_injected_model(self):
        ann = StubAnnotations()
        injected_model = CountingModel()
        record = Record("1", 6000, "A", ["C"])

        scores = make_scorer(ann, 2, 0, model=injected_model).score(record)

        self.assertEqual(len(scores), 2)
        self.assertEqual(injected_model.calls, 1)
        self.assertEqual(ann.model.calls, 0)

    def test_reuses_reference_predictions_for_multiple_alts(self):
        ann = StubAnnotations()
        record = Record("1", 6000, "A", ["C", "G"])

        scores = score_record(record, ann, 2, 0)

        self.assertEqual(
            scores,
            [
                "C|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
                "C|GENE_B|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
                "G|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
                "G|GENE_B|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
            ],
        )
        self.assertEqual(ann.position_data_calls, 2)
        self.assertEqual(ann.model.calls, 1)
        self.assertEqual(ann.model.batch_shapes, [(6, 10005, 4)])

    def test_preserves_insertion_and_deletion_alignment(self):
        insertion_ann = StubAnnotations()
        deletion_ann = StubAnnotations()

        insertion_scores = score_record(
            Record("1", 6000, "A", ["AC"]), insertion_ann, 2, 0
        )
        deletion_scores = score_record(
            Record("1", 6000, "AA", ["A"]), deletion_ann, 2, 0
        )

        self.assertEqual(
            insertion_scores,
            [
                "AC|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
                "AC|GENE_B|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
            ],
        )
        self.assertEqual(
            deletion_scores,
            [
                "A|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
                "A|GENE_B|0.00|0.00|0.00|0.00|-2|-2|-2|-2",
            ],
        )
        self.assertEqual(insertion_ann.model.calls, 1)
        self.assertEqual(deletion_ann.model.calls, 1)
        self.assertEqual(insertion_ann.model.batch_shapes, [(4, 10006, 4)])
        self.assertEqual(deletion_ann.model.batch_shapes, [(4, 10005, 4)])

    def test_preserves_masking_rules(self):
        raw_ann = StubAnnotations()
        masked_ann = StubAnnotations()
        raw_ann.genes = raw_ann.genes[:1]
        masked_ann.genes = masked_ann.genes[:1]
        raw_ann.model = MaskingModel()
        masked_ann.model = MaskingModel()
        record = Record("1", 6000, "A", ["C"])

        self.assertEqual(
            score_record(record, raw_ann, 2, 0),
            ["C|GENE_A|0.80|0.00|0.70|0.00|0|-2|1|-2"],
        )
        self.assertEqual(
            score_record(record, masked_ann, 2, 1),
            ["C|GENE_A|0.00|0.00|0.70|0.00|0|-2|1|-2"],
        )

    def test_skips_inference_for_unsupported_and_complex_alts(self):
        ann = StubAnnotations()
        unsupported = Record(
            "1",
            6000,
            "A",
            [".", "*", "<DEL>", "A[2:321[", "]2:321]A"],
        )
        complex_substitution = Record("1", 6000, "AA", ["CC"])

        self.assertEqual(score_record(unsupported, ann, 2, 0), [])
        self.assertEqual(
            score_record(complex_substitution, ann, 2, 0),
            [
                "CC|GENE_A|.|.|.|.|.|.|.|.",
                "CC|GENE_B|.|.|.|.|.|.|.|.",
            ],
        )
        self.assertEqual(ann.position_data_calls, 0)
        self.assertEqual(ann.model.calls, 0)

    def test_batches_across_records_and_preserves_order(self):
        ann = StubAnnotations()
        ann.genes = ann.genes[:1]
        records = [
            Record("1", 6000, "A", ["C"]),
            Record("1", 6000, "A", ["."]),
            Record("1", 6000, "A", ["G"]),
        ]

        annotated = list(score_records(records, ann, 2, 0, batch_size=4))

        self.assertEqual([record for record, _ in annotated], records)
        self.assertEqual(
            [scores for _, scores in annotated],
            [
                ["C|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2"],
                [],
                ["G|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2"],
            ],
        )
        self.assertEqual(ann.model.batch_shapes, [(4, 10005, 4)])

    def test_limits_batch_size_and_flushes_final_partial_batch(self):
        ann = StubAnnotations()
        ann.genes = ann.genes[:1]
        records = [Record("1", 6000, "A", [alt]) for alt in ("C", "G", "T")]

        list(score_records(records, ann, 2, 0, batch_size=4))

        self.assertEqual(ann.model.batch_shapes, [(4, 10005, 4), (2, 10005, 4)])

    def test_highly_multiallelic_record_keeps_inference_batches_bounded(self):
        ann = StubAnnotations()
        ann.genes = ann.genes[:1]
        alternate_alleles = ["C", "G", "T"] * 40
        record = Record("1", 6000, "A", alternate_alleles)

        scores = make_scorer(ann, 2, 0, batch_size=7).score(record)

        self.assertEqual(len(scores), len(alternate_alleles))
        self.assertTrue(ann.model.batch_shapes)
        self.assertLessEqual(max(shape[0] for shape in ann.model.batch_shapes), 7)

    def test_batches_variable_lengths_across_records(self):
        ann = StubAnnotations()
        ann.genes = ann.genes[:1]
        records = [
            Record("1", 6000, "A", ["AC"]),
            Record("1", 6000, "AA", ["A"]),
        ]

        annotated = list(score_records(records, ann, 2, 0, batch_size=4))

        self.assertEqual(
            [scores for _, scores in annotated],
            [
                ["AC|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2"],
                ["A|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2"],
            ],
        )
        self.assertEqual(ann.model.batch_shapes, [(4, 10006, 4)])

    def test_encodes_and_aligns_variable_length_batch_inputs(self):
        ann = StubAnnotations()
        ann.model = CapturingModel()
        records = [
            Record("1", 6000, "A", ["AC"]),
            Record("1", 6000, "AA", ["A"]),
        ]

        list(score_records(records, ann, 2, 0, batch_size=8))

        inputs = ann.model.inputs[0]
        prepared_tasks = []
        for record in records:
            prepared = make_scorer(ann, 2, 0)._prepare_record(record)
            prepared_tasks.extend(prepared.tasks)

        reverse_tasks = [
            (task_index, task)
            for task_index, task in enumerate(prepared_tasks)
            if task.reverse_output
        ]
        self.assertEqual(
            [inputs.shape[1] - len(task.sequence) for _, task in reverse_tasks],
            [1, 0, 1, 2],
        )
        for task_index, task in enumerate(prepared_tasks):
            with self.subTest(task_index=task_index):
                expected = one_hot_encode(task.sequence)
                padding = inputs.shape[1] - len(task.sequence)
                if task.reverse_output:
                    expected = expected[::-1, ::-1]
                    self.assertFalse(inputs[task_index, :padding].any())
                    actual = inputs[task_index, padding:]
                else:
                    if padding:
                        self.assertFalse(inputs[task_index, -padding:].any())
                    actual = inputs[task_index, : len(task.sequence)]
                np.testing.assert_array_equal(actual, expected)

    def test_rejects_invalid_batch_sizes(self):
        ann = StubAnnotations()

        for batch_size in (0, -1, 1.5, True):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    list(score_records([], ann, 2, 0, batch_size))

    def test_rejects_invalid_distance_and_mask(self):
        ann = StubAnnotations()
        for distance in (-1, 5000, 1.5, True):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                make_scorer(ann, distance, 0)
        for mask in (-1, 2, 0.5, "1"):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                make_scorer(ann, 2, mask)

    def test_missing_reference_chromosome_skips_record(self):
        ann = StubAnnotations()
        record = Record("2", 6000, "A", ["C"])

        with self.assertLogs("spliceai", level="WARNING") as logs:
            self.assertEqual(score_record(record, ann, 2, 0), [])

        self.assertIn("fasta issue", logs.output[0])
        self.assertEqual(ann.model.calls, 0)

    def test_zero_distance_scores_non_deletions(self):
        ann = StubAnnotations()
        ann.genes = ann.genes[:1]

        self.assertEqual(
            make_scorer(ann, 0, 0).score(Record("1", 6000, "A", ["C"])),
            ["C|GENE_A|0.00|0.00|0.00|0.00|0|0|0|0"],
        )
        self.assertEqual(
            make_scorer(ann, 0, 0).score(Record("1", 6000, "A", ["AC"])),
            ["AC|GENE_A|0.00|0.00|0.00|0.00|0|0|0|0"],
        )

    def test_deletion_limit_counts_deleted_bases(self):
        ann = StubAnnotations()
        ann.genes = ann.genes[:1]

        allowed = make_scorer(ann, 1, 0).score(Record("1", 6000, "AAA", ["A"]))
        rejected = make_scorer(ann, 1, 0).score(Record("1", 6000, "AAAA", ["A"]))

        self.assertEqual(
            allowed,
            ["A|GENE_A|0.00|0.00|0.00|0.00|-1|-1|-1|-1"],
        )
        self.assertEqual(rejected, [])
