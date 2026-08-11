import argparse
import logging
import os
import signal
import time
from pathlib import Path

import pysam
import torch
from pyfaidx import Fasta

from spliceai import __version__, logger
from spliceai.annotation import AnnotationFormatError, TranscriptAnnotations
from spliceai.model import EnsembleSpliceAIModel
from spliceai.scoring import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DISTANCE,
    DEFAULT_MASK,
    SplicingScorer,
)

try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out


_PROGRESS_LOG_INTERVAL = 30.0


def configure_process():
    """Configure command-line logging and interrupt handling."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    signal.signal(signal.SIGINT, signal.default_int_handler)


def paths_refer_to_same_file(input_value, output_value):
    """Return whether two path-like CLI values identify the same file."""
    try:
        input_path = os.fspath(input_value)
        output_path = os.fspath(output_value)
    except TypeError:
        return False

    try:
        return os.path.samefile(input_path, output_path)
    except (FileNotFoundError, OSError):
        return Path(input_path).resolve() == Path(output_path).resolve()


def add_spliceai_header(header, overwrite_existing=False):
    """Add the current SpliceAI INFO header under an explicit overwrite policy."""
    if "SpliceAI" in header.info:
        if not overwrite_existing:
            raise ValueError(
                "Input VCF already contains SpliceAI annotations; pass "
                "--overwrite-existing to replace them"
            )
        header.info.remove_header("SpliceAI")

    header.add_line(
        f'##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv{__version__} '
        "variant annotation. These include delta scores (DS) and delta positions "
        "(DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and "
        "donor loss (DL). Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|"
        'DP_AL|DP_DG|DP_DL">'
    )


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def log_scoring_progress(scored_records, interval=_PROGRESS_LOG_INTERVAL):
    """Yield scored records and periodically log completed-record progress."""
    started_at = time.monotonic()
    last_logged_at = started_at
    records_processed = 0

    for scored_record in scored_records:
        yield scored_record
        records_processed += 1
        current_time = time.monotonic()
        if current_time - last_logged_at >= interval:
            elapsed = current_time - started_at
            logger.info(
                "Scored %d records in %.1f seconds (%.1f records/second)",
                records_processed,
                elapsed,
                records_processed / elapsed,
            )
            last_logged_at = current_time

    elapsed = time.monotonic() - started_at
    rate = records_processed / elapsed if elapsed else 0.0
    logger.info(
        "Finished scoring %d records in %.1f seconds (%.1f records/second)",
        records_processed,
        elapsed,
        rate,
    )


def get_options():
    parser = argparse.ArgumentParser(description=f"Version: {__version__}")
    parser.add_argument(
        "-I",
        metavar="input",
        nargs="?",
        default=std_in,
        help="path to the input VCF file, defaults to standard in",
    )
    parser.add_argument(
        "-O",
        metavar="output",
        nargs="?",
        default=std_out,
        help="path to the output VCF file, defaults to standard out",
    )
    parser.add_argument(
        "-R",
        metavar="reference",
        required=True,
        help="path to the reference genome fasta file",
    )
    parser.add_argument(
        "-A",
        metavar="annotation",
        required=True,
        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
        'package), "grch38" (GENCODE V24 canonical annotation file in '
        "package), or path to a similar custom gene annotation file",
    )
    parser.add_argument(
        "-D",
        metavar="distance",
        nargs="?",
        default=DEFAULT_DISTANCE,
        type=int,
        choices=range(5000),
        help="maximum distance between the variant and gained/lost splice "
        f"site, defaults to {DEFAULT_DISTANCE}",
    )
    parser.add_argument(
        "-M",
        metavar="mask",
        nargs="?",
        default=DEFAULT_MASK,
        type=int,
        choices=[0, 1],
        help="mask scores representing annotated acceptor/donor gain and "
        f"unannotated acceptor/donor loss, defaults to {DEFAULT_MASK}",
    )
    parser.add_argument(
        "-B",
        "--batch-size",
        metavar="batch_size",
        default=DEFAULT_BATCH_SIZE,
        type=positive_int,
        help="maximum number of model inputs per inference batch, defaults to "
        f"{DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--threads",
        metavar="threads",
        default=None,
        type=positive_int,
        help="number of PyTorch CPU inference threads, defaults to PyTorch's "
        "current setting",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device, defaults to automatic CUDA detection with CPU fallback",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="replace existing SpliceAI header and record annotations",
    )
    args = parser.parse_args()

    # Set PyTorch threads
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    return args


def main():
    configure_process()
    args = get_options()

    if None in [args.I, args.O, args.D, args.M]:
        logger.error(
            "Usage: spliceai [-h] [-I [input]] [-O [output]] "
            "-R reference -A annotation "
            "[-D [distance]] [-M [mask]] [-B batch_size] [--threads threads] "
            "[--device {auto,cpu,cuda}] [--overwrite-existing]"
        )
        return 2
    if paths_refer_to_same_file(args.I, args.O):
        logger.error("Input and output must refer to different files")
        return 2

    vcf = None
    output = None
    ref_fasta = None
    try:
        vcf = pysam.VariantFile(args.I)
        header = vcf.header
        add_spliceai_header(header, args.overwrite_existing)
        logger.info("Loading models")
        model = EnsembleSpliceAIModel().to_device(args.device)
        logger.info("Parsing transcript annotations")
        transcript_annotations = TranscriptAnnotations(args.A)
        logger.info("Loading reference FASTA")
        ref_fasta = Fasta(args.R, rebuild=False)
        scorer = SplicingScorer(
            model=model,
            transcript_annotations=transcript_annotations,
            ref_fasta=ref_fasta,
            distance=args.D,
            mask=args.M,
            batch_size=args.batch_size,
        )
        output = pysam.VariantFile(args.O, mode="w", header=header)

        logger.info("Initializing scoring")
        for record, scores in log_scoring_progress(scorer.score_batch(vcf)):
            if args.overwrite_existing and "SpliceAI" in record.info:
                del record.info["SpliceAI"]
            if scores:
                record.info["SpliceAI"] = scores
            output.write(record)
    except (AnnotationFormatError, OSError, ValueError) as error:
        logger.error(error)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    finally:
        if ref_fasta is not None:
            ref_fasta.close()
        if vcf is not None:
            vcf.close()
        if output is not None:
            output.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
