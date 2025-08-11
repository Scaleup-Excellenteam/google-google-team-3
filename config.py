import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(ROOT_DIR, "Archive")

DB_NAME = "sentences.db"
DB_PATH = os.path.join(ROOT_DIR, DB_NAME)

JSON_PATH = os.path.join(ROOT_DIR, "sentences.json")
INDEX_PATH = os.path.join(ROOT_DIR, "sentences.idx")
