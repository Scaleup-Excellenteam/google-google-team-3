import json
from Levenshtein import distance, editops
from AutoCompleteData import AutoCompleteData


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_score(user_input, candidate):
    base_score = len(user_input) * 2
    if user_input == candidate:
        return base_score

    ops = editops(user_input, candidate)
    if len(ops) != 1:
        return base_score

    op, src, tgt = ops[0]
    pos = src if op != 'insert' else tgt

    deduction = 0
    if op == 'replace':
        # First char – 5 points, second char – 4, third char – 3, fourth char – 2, fifth and onward – 1 point each
        if pos == 0:
            deduction = 5
        elif pos == 1:
            deduction = 4
        elif pos == 2:
            deduction = 3
        elif pos == 3:
            deduction = 2
        else:
            deduction = 1
    elif op in ('insert', 'delete'):
        # First char – 10 points, second char – 8, third char – 6, fourth char – 4, fifth and onward – 2 points
        if pos == 0:
            deduction = 10
        elif pos == 1:
            deduction = 8
        elif pos == 2:
            deduction = 6
        elif pos == 3:
            deduction = 4
        else:
            deduction = 2

    return base_score - deduction


def get_k_best_completions_bruteforce(json_data, user_input, k=5):
    results = []
    seen = set()

    for entry in json_data:
        if entry['sentence'] in seen:
            continue
        seen.add(entry['sentence'])

        cleaned = entry['cleaned_sentence']

        # Check exact substring match
        if user_input in cleaned:
            score = len(user_input) * 2
            start_index = cleaned.find(user_input)
            results.append(AutoCompleteData(entry['sentence'], user_input, start_index, score))
            continue

        # Check for whole-sentence fuzzy match (distance 1)
        dist = distance(user_input, cleaned)
        if dist == 1:
            score = calculate_score(user_input, cleaned)
            start_index = 0
            results.append(AutoCompleteData(entry['sentence'], user_input, start_index, score))
            continue

        # Check for fuzzy substring match (distance 1)
        # This is where the core logic for fuzzy substrings needs to be more reliable
        for i in range(len(cleaned) - len(user_input) + 1):
            substring = cleaned[i:i + len(user_input)]
            dist = distance(user_input, substring)
            if dist == 1:
                # The substring is a distance 1 match.
                # Now we calculate the score based on this match.
                score = calculate_score(user_input, substring)
                results.append(AutoCompleteData(entry['sentence'], user_input, i, score))
                break  # Move to the next entry after finding one match

    results.sort(key=lambda x: (-x.score, x.completed_sentence))
    return results[:k]


# Example usage:
if __name__ == "__main__":
    data = load_json("sentences.json")
    user_input = "Introduction so"
    matches = get_k_best_completions_bruteforce(data, user_input)

    for m in matches:
        print("Sentence:", m.completed_sentence, "Score:", m.score, "Offset:", m.offset)
