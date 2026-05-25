"""Central config — env-driven so the same code runs in Colab and locally."""

import os

# Checkpoint paths. Defaults assume Colab + Drive-mounted causalsense folder.
ST0_CKPT = os.environ.get(
    "ST0_CKPT", "/content/drive/MyDrive/causalsense/checkpoints/st0_roberta_large"
)
ST1_CKPT = os.environ.get(
    "ST1_CKPT", "/content/drive/MyDrive/causalsense/checkpoints/st1_roberta_large"
)
ST2_CKPT = os.environ.get(
    "ST2_CKPT", "/content/drive/MyDrive/causalsense/checkpoints/st2_bert_large_ner"
)

# Inference settings.
DEVICE = os.environ.get("DEVICE", "auto")  # auto | cuda | mps | cpu

ST0_BATCH = int(os.environ.get("ST0_BATCH", 32))
ST1_BATCH = int(os.environ.get("ST1_BATCH", 32))
ST2_BATCH = int(os.environ.get("ST2_BATCH", 32))
MAX_LEN = int(os.environ.get("MAX_LEN", 128))

# Coref.
COREF_MODEL = os.environ.get("COREF_MODEL", "lingmess")  # lingmess | fcoref

# Resolver overlap threshold (IoU).
SPAN_OVERLAP_THRESHOLD = float(os.environ.get("SPAN_OVERLAP_THRESHOLD", 0.5))

# Neo4j connection (only needed for the neo4j_writer step).
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
