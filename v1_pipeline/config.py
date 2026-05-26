"""Central config — env-driven so the same code runs in Colab and locally."""

import os

# Checkpoint paths. Defaults assume Colab + Drive-mounted causalsense folder.
ST0_CKPT = os.environ.get("ST0_CKPT", "eitang/st0")
ST1_CKPT = os.environ.get("ST1_CKPT", "eitang/st1")
ST2_CKPT = os.environ.get("ST2_CKPT", "eitang/st2")
ST2_BACKEND = os.environ.get("ST2_BACKEND", "rebel")  # rebel | bio
ST2_REBEL_CKPT = os.environ.get("ST2_REBEL_CKPT", "Babelscape/rebel-large")

# Inference settings.
DEVICE = os.environ.get("DEVICE", "auto")  # auto | cuda | mps | cpu

ST0_BATCH = int(os.environ.get("ST0_BATCH", 32))
ST1_BATCH = int(os.environ.get("ST1_BATCH", 32))
ST2_BATCH = int(os.environ.get("ST2_BATCH", 32))
MAX_LEN = int(os.environ.get("MAX_LEN", 128))
ST2_REBEL_MAX_LEN = int(os.environ.get("ST2_REBEL_MAX_LEN", 512))
ST2_REBEL_MAX_NEW_TOKENS = int(os.environ.get("ST2_REBEL_MAX_NEW_TOKENS", 128))

# Coref.
COREF_MODEL = os.environ.get("COREF_MODEL", "lingmess")  # lingmess | fcoref

# Resolver overlap threshold (IoU).
SPAN_OVERLAP_THRESHOLD = float(os.environ.get("SPAN_OVERLAP_THRESHOLD", 0.5))

# Neo4j connection (only needed for the neo4j_writer step).
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
