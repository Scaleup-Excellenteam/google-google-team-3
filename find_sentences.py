import json
import pybktree
from AutoCompleteData import AutoCompleteData
from Levenshtein import distance


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_bk_tree(json_data):
    # Map cleaned_sentence -> original entries
    mapping = {}
    for entry in json_data:
        sent = entry["cleaned_sentence"]
        mapping.setdefault(sent, []).append(entry)
    # Build tree from cleaned sentences
    tree = pybktree.BKTree(distance, mapping.keys())
    return tree, mapping


python
from Levenshtein import editops

class AutoCompleteData:
    def __init__(self, completed_sentence, user_input, start_index, score):
        self.completed_sentence = completed_sentence
        self.user_input = user_input
        self.start_index = start_index
        self.score = score

def calculate_score(user_input, candidate):
    base_score = len(user_input) * 2
    if user_input == candidate:
        return base_score
    ops = editops(user_input, candidate)
    if len(ops) != 1:
        return base_score  # Only handle distance 1
    op, src, tgt = ops[0]
    pos = src if op != 'insert' else tgt
    if op == 'replace':
        deduction = [5, 4, 3, 2] + [1] * (max(len(user_input), len(candidate)) - 4)
        return base_score - deduction[pos] if pos < len(deduction) else base_score - 1
    elif op in ('insert', 'delete'):
        deduction = [10, 8, 6, 4] + [2] * (max(len(user_input), len(candidate)) - 4)
        return base_score - deduction[pos] if pos < len(deduction) else base_score - 2
    return base_score

def get_k_best_completions(tree, mapping, user_input, k=5):
    results = []
    for dist, sentence in tree.find(user_input, 1):
        for entry in mapping[sentence]:
            score = calculate_score(user_input, entry['cleaned_sentence'])
            start_index = entry['cleaned_sentence'].find(user_input)
            results.append(AutoCompleteData(entry['sentence'], user_input, start_index, score))
    results.sort(key=lambda x: (-x.score, x.completed_sentence))
    return results[:k]

# Example usage:
if __name__ == "__main__":
    data = load_json("sentences.json")
    tree, mapping = build_bk_tree(data)

    user_input = "Page 1"
    results = find_close_sentences_bk(tree, mapping, user_input)

    for match in results:
        print(
            f"{match['cleaned_sentence']} (dist={distance(match['cleaned_sentence'], user_input)}) from {match['file_path']}")
