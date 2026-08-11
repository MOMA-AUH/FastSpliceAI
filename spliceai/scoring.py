from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from bx.intervals.intersection import Interval
from pyfaidx import Fasta

from spliceai import logger
from spliceai.annotation import TranscriptAnnotations
from spliceai.model import EnsembleSpliceAIModel, CONTEXT, HALF_CONTEXT
from spliceai.utils import is_valid_allele, normalise_chrom, one_hot_encode

DEFAULT_DISTANCE = 50
DEFAULT_MASK = 0
DEFAULT_BATCH_SIZE = 8

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "SplicingScorer",
]


@dataclass
class _GeneScoreContext:
    gene: Interval
    annotation_distances: tuple[int, int, int]
    reference_prediction: np.ndarray | None = None
    alternate_predictions: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class _RecordScoreContext:
    record: object
    annotations: list[list[str | None]]
    genes: list[_GeneScoreContext]
    pending_predictions: int = 0


@dataclass
class _PredictionTask:
    record_context: _RecordScoreContext
    gene_index: int
    alternate_index: int | None
    inputs: np.ndarray
    output_length: int
    reverse_output: bool


@dataclass
class _PreparedRecord:
    context: _RecordScoreContext
    tasks: Iterator[_PredictionTask]


class SplicingScorer:
    def __init__(
        self,
        model: EnsembleSpliceAIModel,
        transcript_annotations: TranscriptAnnotations,
        ref_fasta: Fasta,
        distance: int = DEFAULT_DISTANCE,
        mask: int = DEFAULT_MASK,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if (
            not isinstance(distance, Integral)
            or isinstance(distance, bool)
            or not 0 <= distance < HALF_CONTEXT
        ):
            raise ValueError("distance must be an integer between 0 and 4999")
        if not isinstance(mask, Integral) or mask not in (0, 1):
            raise ValueError("mask must be 0, 1, or a boolean")
        if (
            not isinstance(batch_size, Integral)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        self.model = model
        self.transcript_annotations = transcript_annotations
        self.ref_fasta = ref_fasta
        self.distance = int(distance)
        self.mask = bool(mask)
        self.batch_size = int(batch_size)
        self.coverage = 2 * self.distance + 1
        self.window_width = CONTEXT + self.coverage

    def score(self, record):
        """Return delta scores for one variant record."""
        _, delta_scores = next(self.score_batch([record]))
        return delta_scores

    def score_batch(self, records):
        """Yield each record and its scores while batching model inputs."""
        contexts = deque()
        pending_tasks = deque()

        for record in records:
            prepared = self._prepare_record(record)
            contexts.append(prepared.context)
            for task in prepared.tasks:
                pending_tasks.append(task)
                if len(pending_tasks) == self.batch_size:
                    self._infer_batch(list(pending_tasks))
                    pending_tasks.clear()
                    yield from self._pop_ready_contexts(contexts)

            # Bound queued records when sparse records prevent a full batch.
            if len(contexts) >= self.batch_size and pending_tasks:
                self._infer_batch(pending_tasks)
                pending_tasks.clear()

            yield from self._pop_ready_contexts(contexts)

        if pending_tasks:
            self._infer_batch(pending_tasks)
        yield from self._pop_ready_contexts(contexts)

    def _make_prediction_task(
        self,
        context,
        gene_index,
        alternate_index,
        sequence,
        reverse_output,
    ):
        inputs = one_hot_encode(sequence)
        if reverse_output:
            inputs = inputs[::-1, ::-1]
        return _PredictionTask(
            record_context=context,
            gene_index=gene_index,
            alternate_index=alternate_index,
            inputs=inputs,
            output_length=len(sequence) - CONTEXT,
            reverse_output=reverse_output,
        )

    def _prepare_record(self, record):
        empty_context = _RecordScoreContext(record, [], [])

        try:
            record.chrom, record.pos, record.ref, len(record.alts)
        except TypeError:
            logger.warning(f"Skipping record (bad input): {record}".strip())
            return _PreparedRecord(empty_context, iter(()))

        genes = self.transcript_annotations.get_overlapping_genes(record.chrom, record.pos)
        if not genes:
            return _PreparedRecord(empty_context, iter(()))

        reference_chrom = next(iter(self.ref_fasta.keys()), "")
        chrom = normalise_chrom(record.chrom, reference_chrom)
        window_start = record.pos - self.window_width // 2 - 1
        window_end = record.pos + self.window_width // 2
        try:
            sequence = self.ref_fasta[chrom][window_start:window_end].seq
        except (IndexError, KeyError, ValueError):
            logger.warning(f"Skipping record (fasta issue): {record}".strip())
            return _PreparedRecord(empty_context, iter(()))

        ref_len = len(record.ref)
        midpoint = self.window_width // 2
        if sequence[midpoint : midpoint + ref_len].casefold() != record.ref.casefold():
            logger.warning(f"Skipping record (ref issue): {record}".strip())
            return _PreparedRecord(empty_context, iter(()))

        if len(sequence) != self.window_width:
            logger.warning(f"Skipping record (near chromosome end): {record}".strip())
            return _PreparedRecord(empty_context, iter(()))

        simple_alt_indexes = []
        annotations = [[None for _ in genes] for _ in record.alts]
        for alt_index, alt in enumerate(record.alts):
            if not is_valid_allele(alt):
                continue
            if not (ref_len > 1 and len(alt) > 1):
                deleted_bases = ref_len - len(alt)
                if deleted_bases > 2 * self.distance:
                    logger.warning(
                        f"Skipping alternate allele (deletion too long): {record}".strip()
                    )
                    continue
                simple_alt_indexes.append(alt_index)
                continue
            for gene_index, gene in enumerate(genes):
                annotations[alt_index][gene_index] = (
                    f"{alt}|{gene.value['name']}|.|.|.|.|.|.|.|."
                )

        context = _RecordScoreContext(record, annotations, [])
        if not simple_alt_indexes:
            return _PreparedRecord(context, iter(()))

        for gene in genes:
            annotation_distances = self.transcript_annotations.get_pos_data(gene, record.pos)
            context.genes.append(_GeneScoreContext(gene, annotation_distances))

        context.pending_predictions = len(genes) * (1 + len(simple_alt_indexes))

        def iter_tasks():
            for gene_index, gene_context in enumerate(context.genes):
                annotation_distances = gene_context.annotation_distances
                pad_start = max(midpoint + annotation_distances[0], 0)
                pad_end = max(midpoint - annotation_distances[1], 0)
                reference_sequence = (
                    "N" * pad_start
                    + sequence[pad_start : self.window_width - pad_end]
                    + "N" * pad_end
                )
                reverse_output = gene_context.gene.strand == "-"
                yield self._make_prediction_task(
                    context, gene_index, None, reference_sequence, reverse_output
                )

                for alt_index in simple_alt_indexes:
                    alternate_sequence = (
                        reference_sequence[:midpoint]
                        + record.alts[alt_index]
                        + reference_sequence[midpoint + ref_len :]
                    )
                    yield self._make_prediction_task(
                        context,
                        gene_index,
                        alt_index,
                        alternate_sequence,
                        reverse_output,
                    )

        return _PreparedRecord(context, iter_tasks())

    def _infer_batch(self, tasks):
        max_length = max(len(task.inputs) for task in tasks)
        model_inputs = np.zeros((len(tasks), max_length, 4), dtype=np.float32)
        for task_index, task in enumerate(tasks):
            if task.reverse_output:
                model_inputs[task_index, -len(task.inputs) :] = task.inputs
            else:
                model_inputs[task_index, : len(task.inputs)] = task.inputs

        predictions = self.model.infer(model_inputs)
        for task_index, task in enumerate(tasks):
            prediction = predictions[task_index]
            if task.reverse_output:
                prediction = prediction[::-1]
            prediction = prediction[: task.output_length][None, :].copy()

            gene_context = task.record_context.genes[task.gene_index]
            if task.alternate_index is None:
                gene_context.reference_prediction = prediction
            else:
                gene_context.alternate_predictions[task.alternate_index] = prediction
            task.record_context.pending_predictions -= 1

    def _format_score(self, record, alternate_index, gene_context):
        alt = record.alts[alternate_index]
        ref_len = len(record.ref)
        alt_len = len(alt)
        score_midpoint = self.coverage // 2
        reference = gene_context.reference_prediction
        alternate = gene_context.alternate_predictions[alternate_index]

        if ref_len > 1:
            deleted_bases = ref_len - alt_len
            alternate = np.concatenate(
                [
                    alternate[:, : score_midpoint + alt_len],
                    np.zeros((1, deleted_bases, 3), dtype=alternate.dtype),
                    alternate[:, score_midpoint + alt_len :],
                ],
                axis=1,
            )
        elif alt_len > 1:
            alternate = np.concatenate(
                [
                    alternate[:, :score_midpoint],
                    np.max(
                        alternate[:, score_midpoint : score_midpoint + alt_len],
                        axis=1,
                    )[:, None, :],
                    alternate[:, score_midpoint + alt_len :],
                ],
                axis=1,
            )

        acceptor_delta = alternate[0, :, 1] - reference[0, :, 1]
        donor_delta = alternate[0, :, 2] - reference[0, :, 2]

        acceptor_gain_index = acceptor_delta.argmax()
        acceptor_loss_index = (-acceptor_delta).argmax()
        donor_gain_index = donor_delta.argmax()
        donor_loss_index = (-donor_delta).argmax()

        acceptor_gain_position = acceptor_gain_index - score_midpoint
        acceptor_loss_position = acceptor_loss_index - score_midpoint
        donor_gain_position = donor_gain_index - score_midpoint
        donor_loss_position = donor_loss_index - score_midpoint

        acceptor_gain = acceptor_delta[acceptor_gain_index]
        acceptor_loss = (
            reference[0, acceptor_loss_index, 1] - alternate[0, acceptor_loss_index, 1]
        )
        donor_gain = donor_delta[donor_gain_index]
        donor_loss = (
            reference[0, donor_loss_index, 2] - alternate[0, donor_loss_index, 2]
        )

        exon_boundary_distance = gene_context.annotation_distances[2]
        acceptor_gain *= not (
            self.mask and acceptor_gain_position == exon_boundary_distance
        )
        acceptor_loss *= not (
            self.mask and acceptor_loss_position != exon_boundary_distance
        )
        donor_gain *= not (self.mask and donor_gain_position == exon_boundary_distance)
        donor_loss *= not (self.mask and donor_loss_position != exon_boundary_distance)

        return (
            f"{alt}|{gene_context.gene.value['name']}|{acceptor_gain:.2f}|"
            f"{acceptor_loss:.2f}|{donor_gain:.2f}|{donor_loss:.2f}|"
            f"{acceptor_gain_position}|{acceptor_loss_position}|"
            f"{donor_gain_position}|{donor_loss_position}"
        )

    def _finalize_scores(self, context):
        delta_scores = []
        for alt_index, annotations in enumerate(context.annotations):
            for gene_index, annotation in enumerate(annotations):
                if annotation is not None:
                    delta_scores.append(annotation)
                    continue

                if gene_index >= len(context.genes):
                    continue
                gene_context = context.genes[gene_index]
                if alt_index in gene_context.alternate_predictions:
                    delta_scores.append(
                        self._format_score(
                            context.record,
                            alt_index,
                            gene_context,
                        )
                    )
        return delta_scores

    def _pop_ready_contexts(self, contexts):
        ready = []
        while contexts and contexts[0].pending_predictions == 0:
            context = contexts.popleft()
            ready.append((context.record, self._finalize_scores(context)))
        return ready
