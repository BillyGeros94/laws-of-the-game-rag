"""Settings shared by the indexing and retrieval stages."""

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "football_laws"
INDEX_DIR = "index"

TOP_K = 4   # how many chunks a question retrieves

# bge was trained so that a question carries this instruction and a passage does not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
