## SpliceAI: A deep learning-based tool to identify splice variants
[![release](https://img.shields.io/badge/release-v2.0.0-orange.svg)](https://img.shields.io/badge/release-v2.0.0-orange.svg)
[![downloads](https://pepy.tech/badge/spliceai)](https://pepy.tech/badge/spliceai)

This package annotates genetic variants with their predicted effect on splicing, as described in [Jaganathan *et al*, Cell 2019 in press](https://doi.org/10.1016/j.cell.2018.12.015). The annotations for all possible substitutions, 1 base insertions, and 1-4 base deletions within genes are available [here](https://basespace.illumina.com/s/otSPW8hnhaZR) for download. These annotations are free for academic and not-for-profit use; other use requires a commercial license from Illumina, Inc.

### License
SpliceAI source code is provided under the [PolyForm Strict License 1.0.0](LICENSE). SpliceAI includes several third party packages provided under other open source licenses, please see [NOTICE](NOTICE) for additional details. The trained models used by SpliceAI (located in this package at spliceai/models) are provided under the [CC BY NC 4.0](spliceai/models/LICENSE) license for academic and non-commercial use; other use requires a commercial license from Illumina, Inc.

Purchase of AI scores and models for commercial use is available at [AI_licensing@illumina.com](mailto:AI_licensing@illumina.com).

### Installation
SpliceAI supports Python 3.10 through 3.13.

SpliceAI can be installed from the [github repository](https://github.com/MOMA-AUH/FastSpliceAI.git):
```sh
pip install git+https://github.com/MOMA-AUH/FastSpliceAI.git
```

SpliceAI uses PyTorch for inference. The five original Keras `.h5` model files
remain bundled and are loaded directly into the PyTorch ensemble; TensorFlow is
not required. By default, command-line scoring selects CUDA when PyTorch reports
it as available and otherwise uses CPU. An explicit `--device cuda` request
fails when CUDA is unavailable unless `--allow-fallback` is also specified.

### Usage
SpliceAI can be run from the command line:
```sh
spliceai -I input.vcf -O output.vcf -R genome.fa -A grch37
# or you can pipe the input and output VCFs
cat input.vcf | spliceai -R genome.fa -A grch37 > output.vcf
```

Required parameters:
 - `-R`: Reference genome fasta file. Can be downloaded from [GRCh37/hg19](http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz) or [GRCh38/hg38](http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz).
 - `-A`: Gene annotation file. Can instead provide `grch37` or `grch38` to use GENCODE V24 canonical annotation files included with the package. To create custom gene annotation files, use `spliceai/annotations/grch37.txt` in repository as template.

Output records contain SpliceAI predictions `ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL` in the INFO column. Only SNVs and simple sequence INDELs containing IUPAC DNA nucleotide codes (REF or ALT is a single base) within genes are annotated; ambiguous codes are encoded as unknown bases. Symbolic, spanning-deletion, missing, and breakend alleles are skipped. Variants in multiple genes have separate predictions for each gene.

Optional parameters:
 - `-I`: Input VCF with variants of interest (default: standard input).
 - `-O`: Output path (default: standard output). The output must not refer to the same file as the input. Completed path outputs atomically replace their destination.
 - `-D`: Maximum distance between the variant and gained/lost splice site (default: 50).
 - `-M`: Mask scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss (default: 0).
 - `--output-type`: Output encoding following bcftools notation: `b` for compressed BCF, `z` for compressed VCF, or `v` for uncompressed VCF (default: `v`). Encoding is independent of the filename, so `.bcf`, `.bcf.gz`, and `.bcf.bgz` are equivalent when used with `b`.
 - `--write-index[=FMT]`: Index compressed path output after scoring. `FMT` may be `csi` (the default when omitted) or `tbi`. CSI supports compressed VCF and BCF; TBI supports compressed VCF only. Indexing requires coordinate-sorted records and cannot be used with uncompressed VCF or standard output.
 - `--overwrite-existing`: Replace an existing SpliceAI header and all per-record SpliceAI values. Without this flag, an input VCF that already contains a SpliceAI header is rejected to prevent mixed-version annotations.
 - `--threads`: Number of CPU threads used by PyTorch inference. By default, PyTorch uses all available CPU cores. To avoid degraded performance when running jobs on HPC nodes, the number of threads should be less than or equal to the number of requested cores/threads.
 - `-B`, `--batch-size`: Maximum number of model inputs evaluated in each inference batch (default: 8). Inputs from multiple variants may share a batch.
 - `--device`: Inference device: `auto`, `cpu`, or `cuda` (default: `auto`). Automatic mode selects CUDA when PyTorch reports it as available and otherwise selects CPU. An unavailable explicit CUDA request fails unless `--allow-fallback` is specified.
 - `--bfloat16`: Use bfloat16 precision for inference. The command fails if bfloat16 autocast is unavailable on the selected device unless `--allow-fallback` is specified.
 - `--allow-fallback`: Fall back to CPU when an explicit CUDA request is unavailable, and to float32 when bfloat16 autocast is unavailable.

For programmatic variant scoring, construct the independent annotation,
reference, and model resources once and reuse a `SplicingScorer`:

```python
from pyfaidx import Fasta

from spliceai.annotation import TranscriptAnnotations
from spliceai.model import EnsembleSpliceAIModel
from spliceai.scoring import SplicingScorer

model = EnsembleSpliceAIModel()
annotations = TranscriptAnnotations("grch37")
reference = Fasta("genome.fa", rebuild=False)
try:
    scorer = SplicingScorer(
        model=model,
        transcript_annotations=annotations,
        ref_fasta=reference,
        distance=50,
        mask=0,
        batch_size=8,
    )
    scores = scorer.score(record)
    annotated_records = scorer.score_batch(records)  # Lazy iterator of records and scores.
finally:
    reference.close()
```

`EnsembleSpliceAIModel` is a `torch.nn.Module`. Its `forward` method accepts a
channels-last tensor with shape `(batch, length, 4)` and returns
`(batch, length - 10000, 3)`. The `infer` method provides the same shapes with a
NumPy input and output.

Details of SpliceAI INFO field:

|    ID    | Description |
| -------- | ----------- |
|  ALLELE  | Alternate allele |
|  SYMBOL  | Gene symbol |
|  DS_AG   | Delta score (acceptor gain) |
|  DS_AL   | Delta score (acceptor loss) |
|  DS_DG   | Delta score (donor gain) |
|  DS_DL   | Delta score (donor loss) |
|  DP_AG   | Delta position (acceptor gain) |
|  DP_AL   | Delta position (acceptor loss) |
|  DP_DG   | Delta position (donor gain) |
|  DP_DL   | Delta position (donor loss) |

Delta score of a variant, defined as the maximum of (DS_AG, DS_AL, DS_DG, DS_DL), ranges from 0 to 1 and can be interpreted as the probability of the variant being splice-altering. In the paper, a detailed characterization is provided for 0.2 (high recall), 0.5 (recommended), and 0.8 (high precision) cutoffs. Delta position conveys information about the location where splicing changes relative to the variant position (positive values are downstream of the variant, negative values are upstream).

### Examples
A sample input file and the corresponding output file can be found at `examples/input.vcf` and `examples/output.vcf` respectively. The output `T|RYR1|0.00|0.00|0.91|0.08|-28|-46|-2|-31` for the variant `19:38958362 C>T` can be interpreted as follows:
* The probability that the position 19:38958360 (=38958362-2) is used as a splice donor increases by 0.91.
* The probability that the position 19:38958331 (=38958362-31) is used as a splice donor decreases by 0.08.

Similarly, the output `CA|TTN|0.07|1.00|0.00|0.00|-7|-1|35|-29` for the variant `2:179415988 C>CA` has the following interpretation:
* The probability that the position 2:179415981 (=179415988-7) is used as a splice acceptor increases by 0.07.
* The probability that the position 2:179415987 (=179415988-1) is used as a splice acceptor decreases by 1.00.

### Frequently asked questions

**1. Why are some variants not scored by SpliceAI?**

SpliceAI only annotates variants within genes defined by the gene annotation file. Additionally, SpliceAI does not annotate variants if they are close to chromosome ends (5kb on either side), deletions of length greater than twice the input parameter ```-D```, or inconsistent with the reference fasta file.

**2. What are the differences between raw (```-M 0```) and masked (```-M 1```) precomputed files?**

The raw files also include splicing changes corresponding to strengthening annotated splice sites and weakening unannotated splice sites, which are typically much less pathogenic than weakening annotated splice sites and strengthening unannotated splice sites. The delta scores of such splicing changes are set to 0 in the masked files. We recommend using raw files for alternative splicing analysis and masked files for variant interpretation.

**3. Can SpliceAI be used to score custom sequences?**

Yes, install SpliceAI and use the following script:  

```python
from spliceai.model import EnsembleSpliceAIModel
from spliceai.encoding import one_hot_encode

input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
# Replace this with your custom sequence

context = 10000
model = EnsembleSpliceAIModel()
# To use CUDA, call model.to("cuda") before inference.
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = model.infer(x)

acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]
```

### Contact
Kishore Jaganathan: kjaganathan@illumina.com
