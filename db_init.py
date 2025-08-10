import sqlite3
import re
import os

DB_NAME = 'sentences.db'
FILES_DIR = 'Archive'


def create_database():
    """Connects to the SQLite database and creates the 'sentences' table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentences (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            cleaned_sentence TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            file_path TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()


def clean_sentence(sentence):
    """Cleans the sentence by removing unnecessary whitespace and punctuation."""
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)  # Keep only alphabet, spaces and digits

    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    return sentence.strip()


def insert_sentences_from_file(file_path):
    """Reads a text file, splits it into sentences, and inserts them into the database."""
    if not os.path.exists(file_path):
        print(f"Error: The file at '{file_path}' was not found.")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sentences = content.split('\n')

            line_number = 1
            for sentence in sentences:
                cleaned_sentence = clean_sentence(sentence)
                if cleaned_sentence:
                    cursor.execute('''
                        INSERT INTO sentences (sentence, cleaned_sentence, line_number, file_path)
                        VALUES (?, ?, ?, ?);
                    ''', (sentence, cleaned_sentence, line_number, file_path))
                line_number += 1

        conn.commit()
        print(f"Successfully processed and inserted sentences from '{file_path}'.")

    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {e}")
    finally:
        conn.close()


def insert_all_files():
    files_dir = os.path.join(os.getcwd(), FILES_DIR)
    for root, _, files in os.walk(files_dir):
        for file in files:
            file_path = os.path.join(root, file)
            insert_sentences_from_file(file_path)


if __name__ == '__main__':
    create_database()

    insert_all_files()
    print("Database initialization complete.")
