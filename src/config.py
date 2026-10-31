"""Settings shared by the indexing, retrieval and generation stages."""

# --- Indexing and retrieval ---
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "football_laws"
INDEX_DIR = "index"

TOP_K = 4   # how many chunks a question retrieves

# bge was trained so that a question carries this instruction and a passage does not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Generation ---
# Any OpenAI-compatible provider works. To switch, change these three lines.
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_API_KEY_ENV = "GROQ_API_KEY"   # name of the environment variable that holds the key

LLM_TEMPERATURE = 0
MAX_ANSWER_TOKENS = 1500

# Refuse without calling the LLM when the best chunk is farther than this (cosine distance).
# PROVISIONAL: set from five test questions (best hits of answerable questions 0.15-0.27,
# the question with no answer 0.38). Calibrate it in the evaluation. None disables the check.
MAX_TOP1_DISTANCE = 0.32

REFUSAL = "The provided documents do not contain this."
