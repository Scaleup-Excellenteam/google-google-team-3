import os
import re
import json

FILES_DIR = 'Archive'

def clean_sentence(sentence):
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()

def json_init(output_json_path):
    files_dir = os.path.join(os.getcwd(), FILES_DIR)
    all_sentences = []
    for root, _, files in os.walk(files_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, sentence in enumerate(f, start=1):
                    cleaned_sentence = clean_sentence(sentence)
                    if cleaned_sentence:
                        all_sentences.append({
                            "sentence": sentence.strip(),
                            "cleaned_sentence": cleaned_sentence,
                            "line_number": line_number,
                            "file_path": file_path
                        })
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    json_init('sentences.json')
    print("JSON initialization complete.")