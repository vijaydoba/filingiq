import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
CLAUDE_MODEL = "claude-sonnet-5"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
