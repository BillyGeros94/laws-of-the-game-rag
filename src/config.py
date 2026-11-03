"""Settings shared by the indexing, retrieval and generation stages."""

# --- Indexing and retrieval ---
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "football_laws"
INDEX_DIR = "index"

# bge was trained so that a question carries this instruction and a passage does not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# How retrieval works: "dense" (embedding search only, as in v1), "hybrid" (embedding search
# plus BM25 keyword search, merged by reciprocal rank fusion) or "grade" (hybrid, then a
# language model grades the candidates; src/grade.py). The evaluation can compare them:
# eval.py --mode.
RETRIEVAL_MODE = "grade"
TOP_K = 4         # how many passages dense and hybrid mode return (and grade mode's fallback)
CANDIDATES = 20   # how many results each search contributes to the fusion
RRF_K = 60        # reciprocal rank fusion constant (the usual value)

# Grade mode: the grader sees the GRADE_POOL best search results, each shortened to
# GRADE_PASSAGE_CHARS (Groq's free tier allows 8,000 tokens a minute per model; this keeps a
# grading call near 2,500 tokens). Every passage graded GRADE_MIN or more (on 0-3) is kept, up
# to MAX_SOURCES. With none, or if grading fails, the top TOP_K of the hybrid search are used.
GRADE_MODEL = "openai/gpt-oss-20b"
GRADE_POOL = 12
GRADE_PASSAGE_CHARS = 600
GRADE_MIN = 2
MAX_SOURCES = 6

# --- Generation ---
# Any OpenAI-compatible provider works. To switch, change these three lines.
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_API_KEY_ENV = "GROQ_API_KEY"   # name of the environment variable that holds the key

LLM_TEMPERATURE = 0
MAX_ANSWER_TOKENS = 1500

# The gate: refuse without calling the answer model when nothing retrieved is close to the
# question. A passage graded GRADE_MIN or more always passes. Otherwise the closest passage
# must be within this cosine distance, which only catches clearly off-topic questions
# (answerable questions reach 0.31, unanswerable ones start at 0.21); the answer model
# refuses the rest. None disables the check.
MAX_TOP1_DISTANCE = 0.32

REFUSAL = "The provided documents do not contain this."
