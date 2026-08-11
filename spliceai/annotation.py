from bisect import bisect_left
from csv import DictReader
from csv import Error as CSVError
from importlib.resources import files

from bx.intervals.intersection import Interval, IntervalTree

from spliceai import name
from spliceai.utils import normalise_chrom

__all__ = ["AnnotationFormatError", "TranscriptAnnotations"]

_NAME_COLUMN = "#NAME"
_CHROM_COLUMN = "CHROM"
_STRAND_COLUMN = "STRAND"
_TX_START_COLUMN = "TX_START"
_TX_END_COLUMN = "TX_END"
_EXON_START_COLUMN = "EXON_START"
_EXON_END_COLUMN = "EXON_END"
_REQUIRED_COLUMNS = {
    _NAME_COLUMN,
    _CHROM_COLUMN,
    _STRAND_COLUMN,
    _TX_START_COLUMN,
    _TX_END_COLUMN,
    _EXON_START_COLUMN,
    _EXON_END_COLUMN,
}


class AnnotationFormatError(ValueError):
    """Raised when a gene annotation file does not match the expected format."""


def _parse_coordinates(value, field, path, line_number):
    try:
        return [int(item) for item in value.strip(",").split(",") if item]
    except (AttributeError, ValueError) as error:
        raise AnnotationFormatError(
            f"{path}:{line_number}: {field} must be a comma-separated integer list"
        ) from error


def _format_transcript_interval(row, order, path, line_number):
    name = row[_NAME_COLUMN]
    chrom = row[_CHROM_COLUMN]
    strand = row[_STRAND_COLUMN]
    if not name or not chrom:
        raise AnnotationFormatError(
            f"{path}:{line_number}: gene name and chromosome are required"
        )
    if strand not in {"+", "-"}:
        raise AnnotationFormatError(f"{path}:{line_number}: STRAND must be '+' or '-'")

    try:
        tx_start = int(row[_TX_START_COLUMN])
        tx_end = int(row[_TX_END_COLUMN])
    except (TypeError, ValueError) as error:
        raise AnnotationFormatError(
            f"{path}:{line_number}: transcript coordinates must be integers"
        ) from error
    if tx_start < 0 or tx_start >= tx_end:
        raise AnnotationFormatError(
            f"{path}:{line_number}: transcript coordinates are invalid"
        )

    exon_starts = _parse_coordinates(
        row[_EXON_START_COLUMN], _EXON_START_COLUMN, path, line_number
    )
    exon_ends = _parse_coordinates(
        row[_EXON_END_COLUMN], _EXON_END_COLUMN, path, line_number
    )
    if not exon_starts or len(exon_starts) != len(exon_ends):
        raise AnnotationFormatError(
            f"{path}:{line_number}: exon coordinate lists must be nonempty "
            "and have equal lengths"
        )

    boundaries = []
    for exon_start, exon_end in zip(exon_starts, exon_ends):
        if exon_start < tx_start or exon_start >= exon_end or exon_end > tx_end:
            raise AnnotationFormatError(
                f"{path}:{line_number}: exon coordinates are invalid"
            )
        boundaries.extend((exon_start + 1, exon_end))

    return Interval(
        tx_start,
        tx_end,
        strand=strand,
        value={
            "name": name,
            "exon_boundaries": tuple(sorted(set(boundaries))),
            "order": order,
        },
    )


class TranscriptAnnotations(dict[str, IntervalTree]):
    """Index transcript annotations by chromosome."""

    def __init__(self, annotations):
        super().__init__()

        if annotations == "grch37":
            annotations = files(name).joinpath("annotations/grch37.txt")
        elif annotations == "grch38":
            annotations = files(name).joinpath("annotations/grch38.txt")

        try:
            with (
                annotations.open("r", newline="")
                if hasattr(annotations, "open")
                else open(annotations, newline="")
            ) as annotation_file:
                rows = DictReader(annotation_file, delimiter="\t")
                missing = _REQUIRED_COLUMNS.difference(rows.fieldnames or ())
                if missing:
                    columns = ", ".join(sorted(missing))
                    raise AnnotationFormatError(
                        f"{annotations}: missing required columns: {columns}"
                    )

                for order, row in enumerate(rows):
                    if row[_CHROM_COLUMN] not in self:
                        self[row[_CHROM_COLUMN]] = IntervalTree()
                    self[row[_CHROM_COLUMN]].insert_interval(
                        _format_transcript_interval(
                            row, order, annotations, rows.line_num
                        )
                    )
        except CSVError as error:
            raise AnnotationFormatError(
                f"{annotations}: unable to parse tab-separated data: {error}"
            ) from error

    def get_overlapping_genes(self, chrom, pos) -> list[Interval]:
        annotation_chrom = next(iter(self), "")
        chrom = normalise_chrom(chrom, annotation_chrom)
        if (tree := self.get(chrom)) is None:
            return []
        return sorted(tree.find(pos - 1, pos), key=lambda gene: gene.value["order"])

    def get_pos_data(self, gene: Interval, pos) -> tuple[int, int, int]:
        dist_tx_start = gene.start + 1 - pos
        dist_tx_end = gene.end - pos

        boundaries = gene.value["exon_boundaries"]
        boundary_index = bisect_left(boundaries, pos)
        distances = []
        if boundary_index:
            distances.append(boundaries[boundary_index - 1] - pos)
        if boundary_index < len(boundaries):
            distances.append(boundaries[boundary_index] - pos)
        dist_exon_bdry = min(distances, key=lambda distance: (abs(distance), distance))
        return dist_tx_start, dist_tx_end, dist_exon_bdry
