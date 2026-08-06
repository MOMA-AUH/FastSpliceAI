import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spliceai.utils import Annotator

ANNOTATIONS = """#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END
GENE_A\t1\t+\t9\t20\t9,13,\t12,20,
GENE_B\t1\t-\t4\t20\t4,\t20,
SINGLE_BASE\t1\t+\t30\t31\t30,\t31,
LONG_INTRON\t1\t+\t40\t10000\t40,9990,\t50,10000,
"""


class TestAnnotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        annotations = Path(cls.temp_dir.name) / "annotations.txt"
        annotations.write_text(ANNOTATIONS)

        with (
            patch("spliceai.utils.EnsembleModel", return_value=object()),
            patch("spliceai.utils.Fasta", return_value={}),
        ):
            cls.ann = Annotator("reference.fa", annotations)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def get_gene(self, name, pos):
        return next(
            gene
            for gene in self.ann.get_overlapping_genes("1", pos)
            if gene.value["name"] == name
        )

    def test_gene_lookup_includes_one_based_transcript_endpoints(self):
        gene_intervals = self.ann.get_overlapping_genes("1", 10)
        self.assertEqual(
            [gene.value["name"] for gene in gene_intervals], ["GENE_A", "GENE_B"]
        )
        self.assertEqual([gene.strand for gene in gene_intervals], ["+", "-"])

        gene_intervals = self.ann.get_overlapping_genes("1", 20)
        self.assertEqual(
            [gene.value["name"] for gene in gene_intervals], ["GENE_A", "GENE_B"]
        )

        self.assertEqual(self.ann.get_overlapping_genes("1", 21), [])

    def test_gene_lookup_normalises_prefix_without_adding_unknown_chromosomes(self):
        gene_intervals = self.ann.get_overlapping_genes("chr1", 10)
        self.assertEqual(
            [gene.value["name"] for gene in gene_intervals], ["GENE_A", "GENE_B"]
        )

        chromosomes = set(self.ann.genes)
        self.assertEqual(self.ann.get_overlapping_genes("2", 10), [])
        self.assertEqual(set(self.ann.genes), chromosomes)

    def test_position_data_preserves_boundary_distances_and_tie_breaking(self):
        gene = self.get_gene("GENE_A", 13)
        self.assertEqual(self.ann.get_pos_data(gene, 13), (-3, 7, -1))
        self.assertEqual(self.ann.get_pos_data(gene, 10), (0, 10, 0))
        self.assertEqual(self.ann.get_pos_data(gene, 20), (-10, 0, 0))

    def test_single_base_exon_has_one_boundary_position(self):
        gene = self.get_gene("SINGLE_BASE", 31)
        self.assertEqual(self.ann.get_pos_data(gene, 31), (0, 0, 0))

    def test_exon_lookup_searches_beyond_default_max_distance(self):
        gene = self.get_gene("LONG_INTRON", 5000)
        self.assertEqual(self.ann.get_pos_data(gene, 5000), (-4959, 5000, -4950))


if __name__ == "__main__":
    unittest.main()
