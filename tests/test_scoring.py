import unittest
from collections import namedtuple
from importlib.resources import files
from types import SimpleNamespace

import torch
from pyfaidx import Fasta

from spliceai.annotation import TranscriptAnnotations
from spliceai.model import EnsembleSpliceAIModel
from spliceai.scoring import DEFAULT_BATCH_SIZE, SplicingScorer
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

    def __call__(self, inputs):
        self.calls += 1
        self.batch_shapes.append(inputs.shape)
        output_length = inputs.shape[1] - 10000
        return torch.zeros((inputs.shape[0], output_length, 3), dtype=torch.float32)


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
        cls.ann = TranscriptAnnotations("grch37")
        cls.ann.model = EnsembleSpliceAIModel()
        cls.ann.ref_fasta = Fasta(fasta_path, rebuild=False)

    @classmethod
    def tearDownClass(cls):
        cls.ann.ref_fasta.close()

    def test_reference_outputs_remain_stable(self):
        records = [
            Record("10", 94077, "A", ["C"]),
            Record("chr10", 94555, "C", ["T"]),
            Record("10", 94077, "A", ["AC"]),
            Record("10", 94077, "AC", ["A"]),
        ]

        raw_scores = list(score_records(records, self.ann, 500, 0, 8))
        masked_scores = list(score_records(records[2:], self.ann, 500, 1, 4))

        self.assertEqual(
            [scores for _, scores in raw_scores],
            [
                ["C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193"],
                ["T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0"],
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


class TestScoringPipeline(unittest.TestCase):
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

    def test_batches_records_without_changing_order(self):
        ann = StubAnnotations()
        ann.genes = ann.genes[:1]
        records = [Record("1", 6000, "A", [alt]) for alt in ("C", "G", "T")]

        annotated = list(score_records(records, ann, 2, 0, batch_size=4))

        self.assertEqual([record for record, _ in annotated], records)
        self.assertEqual(
            [scores for _, scores in annotated],
            [
                ["C|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2"],
                ["G|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2"],
                ["T|GENE_A|0.00|0.00|0.00|0.00|-2|-2|-2|-2"],
            ],
        )
        self.assertEqual(ann.model.batch_shapes, [(4, 10005, 4), (2, 10005, 4)])

    def test_skips_unsupported_alleles_without_inference(self):
        ann = StubAnnotations()
        record = Record("1", 6000, "A", [".", "*", "<DEL>", "A[2:321["])

        self.assertEqual(score_record(record, ann, 2, 0), [])
        self.assertEqual(ann.position_data_calls, 0)
        self.assertEqual(ann.model.calls, 0)


if __name__ == "__main__":
    unittest.main()
