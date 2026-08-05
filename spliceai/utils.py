import sys
from collections import defaultdict
from importlib.resources import files

import numpy as np
import pandas as pd
from bx.intervals.intersection import Interval, IntervalTree
from keras.models import load_model
from pyfaidx import Fasta

from spliceai import logger, name

_ONE_HOT_ENCODING = np.zeros((256, 4), dtype=np.int64)
_ONE_HOT_ENCODING[[ord(base) for base in "Aa"], 0] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Cc"], 1] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Gg"], 2] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Tt"], 3] = 1


class Annotator:
    def __init__(self, ref_fasta, annotations):
        if annotations == "grch37":
            annotations = files(name).joinpath("annotations/grch37.txt")
        elif annotations == "grch38":
            annotations = files(name).joinpath("annotations/grch38.txt")

        self.genes = defaultdict(IntervalTree)
        try:
            df = pd.read_csv(annotations, sep="\t", dtype={"CHROM": object})
            for idx, (_, row) in enumerate(df.iterrows()):
                exons = IntervalTree()
                exon_starts = (
                    int(x) for x in row["EXON_START"].strip(",").split(",") if x
                )
                exon_ends = (int(x) for x in row["EXON_END"].strip(",").split(",") if x)
                for exon_start, exon_end in zip(exon_starts, exon_ends):
                    exons.insert_interval(Interval(exon_start, exon_end))

                self.genes[row["CHROM"]].insert_interval(
                    Interval(
                        int(row["TX_START"]),
                        int(row["TX_END"]),
                        strand=row["STRAND"],
                        value={
                            "name": row["#NAME"],
                            "exons": exons,
                            "order": idx,
                        },
                    )
                )
        except OSError as e:
            logger.error(e)
            sys.exit()
        except (KeyError, pd.errors.ParserError) as e:
            logger.error(
                f"Gene annotation file {annotations} not formatted properly: {e}"
            )
            sys.exit()

        try:
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
        except OSError as e:
            logger.error(e)
            sys.exit()

        paths = (f"models/spliceai{x}.h5" for x in range(1, 6))
        self.models = [load_model(files(name).joinpath(x)) for x in paths]

    def get_overlapping_genes(self, chrom, pos) -> list[Interval]:
        annotation_chrom = next(iter(self.genes), "")
        chrom = normalise_chrom(chrom, annotation_chrom)
        if (tree := self.genes.get(chrom)) is None:
            return []
        return sorted(tree.find(pos - 1, pos), key=lambda gene: gene.value["order"])

    def get_pos_data(self, gene: Interval, pos) -> tuple[int, int, int]:
        dist_tx_start = gene.start + 1 - pos
        dist_tx_end = gene.end - pos

        exons = gene.value["exons"]
        distances = [
            boundary - pos
            for exon in exons.find(pos - 1, pos)
            for boundary in (exon.start + 1, exon.end)
        ]

        max_dist = gene.end - gene.start + 1
        if before := exons.before(pos, max_dist=max_dist, num_intervals=1):
            distances.append(before[0].end - pos)
        if after := exons.after(pos - 1, max_dist=max_dist, num_intervals=1):
            distances.append(after[0].start + 1 - pos)

        dist_exon_bdry = min(distances, key=lambda distance: (abs(distance), distance))

        return dist_tx_start, dist_tx_end, dist_exon_bdry


def one_hot_encode(seq):
    sequence = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _ONE_HOT_ENCODING[sequence]


def normalise_chrom(source, target):
    def has_prefix(x):
        return x.startswith("chr")

    if has_prefix(source) and not has_prefix(target):
        return source.strip("chr")
    elif not has_prefix(source) and has_prefix(target):
        return "chr" + source

    return source


def get_delta_scores(record, ann, dist_var, mask):
    cov = 2 * dist_var + 1
    wid = 10000 + cov
    delta_scores = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logger.warning(f"Skipping record (bad input): {record}")
        return delta_scores

    genes = ann.get_overlapping_genes(record.chrom, record.pos)
    if not genes:
        return delta_scores

    chrom = normalise_chrom(record.chrom, next(iter(ann.ref_fasta.keys())))
    try:
        seq = ann.ref_fasta[chrom][
            record.pos - wid // 2 - 1 : record.pos + wid // 2
        ].seq
    except (IndexError, ValueError):
        logger.warning(f"Skipping record (fasta issue): {record}")
        return delta_scores

    if seq[wid // 2 : wid // 2 + len(record.ref)].upper() != record.ref:
        logger.warning(f"Skipping record (ref issue): {record}")
        return delta_scores

    if len(seq) != wid:
        logger.warning(f"Skipping record (near chromosome end): {record}")
        return delta_scores

    if len(record.ref) > 2 * dist_var:
        logger.warning(f"Skipping record (ref too long): {record}")
        return delta_scores

    for j in range(len(record.alts)):
        for gene in genes:
            if "." in record.alts[j] or "-" in record.alts[j] or "*" in record.alts[j]:
                continue

            if "<" in record.alts[j] or ">" in record.alts[j]:
                continue

            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                delta_scores.append(
                    "{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], gene.value["gene"])
                )
                continue

            dist_ann = ann.get_pos_data(gene, record.pos)
            pad_size = [max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len - alt_len, 0)

            x_ref = (
                "N" * pad_size[0]
                + seq[pad_size[0] : wid - pad_size[1]]
                + "N" * pad_size[1]
            )
            x_alt = (
                x_ref[: wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len :]
            )

            x_ref = one_hot_encode(x_ref)[None, :]
            x_alt = one_hot_encode(x_alt)[None, :]

            if gene.strand == "-":
                x_ref = x_ref[:, ::-1, ::-1]
                x_alt = x_alt[:, ::-1, ::-1]

            y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(5)], axis=0)
            y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(5)], axis=0)

            if gene.strand == "-":
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]

            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate(
                    [
                        y_alt[:, : cov // 2 + alt_len],
                        np.zeros((1, del_len, 3)),
                        y_alt[:, cov // 2 + alt_len :],
                    ],
                    axis=1,
                )
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate(
                    [
                        y_alt[:, : cov // 2],
                        np.max(y_alt[:, cov // 2 : cov // 2 + alt_len], axis=1)[
                            :, None, :
                        ],
                        y_alt[:, cov // 2 + alt_len :],
                    ],
                    axis=1,
                )

            y = np.concatenate([y_ref, y_alt])

            idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()

            mask_pa = np.logical_and((idx_pa - cov // 2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na - cov // 2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd - cov // 2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd - cov // 2 != dist_ann[2]), mask)

            delta_scores.append(
                "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
                    record.alts[j],
                    gene.value["name"],
                    (y[1, idx_pa, 1] - y[0, idx_pa, 1]) * (1 - mask_pa),
                    (y[0, idx_na, 1] - y[1, idx_na, 1]) * (1 - mask_na),
                    (y[1, idx_pd, 2] - y[0, idx_pd, 2]) * (1 - mask_pd),
                    (y[0, idx_nd, 2] - y[1, idx_nd, 2]) * (1 - mask_nd),
                    idx_pa - cov // 2,
                    idx_na - cov // 2,
                    idx_pd - cov // 2,
                    idx_nd - cov // 2,
                )
            )

    return delta_scores
