import sys
import os

from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.g2p.generator import PyniniGenerator
from montreal_forced_aligner.validation import PretrainedValidator


corpus = sys.argv[1]
out = corpus + "_aligned"

if not os.path.exists(out):
    os.mkdir(out)

# validation sublimation
print("validating corpus...")
validator = PretrainedValidator(
    corpus_directory=corpus,
    dictionary_path="english_us_arpa",
    acoustic_model_path="english_us_arpa",
)
validator.setup()
validator.validate()

oovs = validator.oovs
if oov_words:
    print(f"found {len(oovs)} oovs: {oovs}")

validator.cleanup()

# align with the divine
print("aligning...")
aligner = PretrainedAligner(
    corpus_directory=corpus,
    dictionary_path="english_us_arpa",
    acoustic_model_path="english_us_arpa",
    output_directory=out,
)

# g2p gameplay
aligner.g2p_model_path = "english_us_arpa"

aligner.setup()
aligner.align()
aligner.export_files(output_directory=out)
aligner.cleanup()

print(f"aligned TextGrids saved to {out}")
