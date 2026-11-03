"""Settings shared by the indexing, retrieval and generation stages."""

# --- Indexing and retrieval ---
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "football_laws"
INDEX_DIR = "index"

TOP_K = 4   # how many chunks a question retrieves

# How retrieval works: "dense" (embedding search only, v1), "hybrid" (embedding search plus
# BM25 keyword search, merged by reciprocal rank fusion) or "rerank" (hybrid, then a
# cross-encoder rescores the candidates). The evaluation can compare them: eval.py --mode.
RETRIEVAL_MODE = "rerank"
CANDIDATES = 20   # how many results each search contributes before fusion and reranking
RRF_K = 60        # reciprocal rank fusion constant (the usual value)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # small enough for a free host

# bge was trained so that a question carries this instruction and a passage does not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Generation ---
# Any OpenAI-compatible provider works. To switch, change these three lines.
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_API_KEY_ENV = "GROQ_API_KEY"   # name of the environment variable that holds the key

LLM_TEMPERATURE = 0
MAX_ANSWER_TOKENS = 1500

# The gate: refuse without calling the LLM when nothing retrieved is relevant enough.
# In rerank mode it uses the cross-encoder's score of the best chunk. On the evaluation set
# answerable questions score 3.4-8.9 and unanswerable ones -6.0 to 2.6, so 2.0 sits below
# the gap: a wrongly refused question costs more than one the model refuses itself.
# Set from the same 26 questions it is measured on; check it on new ones. None disables it.
MIN_RERANK_SCORE = 2.0
# In dense and hybrid mode it uses cosine distance, which separates the two poorly
# (answerable 0.09-0.31, unanswerable 0.21-0.38) and only catches clearly off-topic questions.
MAX_TOP1_DISTANCE = 0.32

REFUSAL = "The provided documents do not contain this."
