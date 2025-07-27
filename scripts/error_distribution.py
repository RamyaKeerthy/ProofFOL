import os
import re
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict


def err_cat(x):
    if x == '' or x is None:
        return "Not detectable"
    elif 'NoneType' in x:
        return "Type"
    elif 'Unexpected token' in x:
        return "Token"
    elif 'multiple arities' in x:
        return "Arities"
    else:
        return "Parsing"


def extract_file_info(directory):
    # Known sets
    datasets = ['folio', 'prontoqa', 'proverqa', 'proof']
    models = ['phi4b', 'gemma4b', 'qwen3b']
    methods = ['FOLIO', 'Mixed', 'ProofFOL']
    sizes = {'folio': 203, 'proof': 600, 'prontoqa': 500, 'proverqa': 500}
    valid_labels = ['True', 'False', 'Unknown']

    # Regex for finetuned files
    pattern_finetuned = re.compile(
        rf'^({"|".join(datasets)})_({"|".join(models)})({"|".join(methods)})_fol_finetuned-tool\.json$'
    )
    # Regex for 3shot files
    pattern_3shot = re.compile(
        rf'^({"|".join(datasets)})_({"|".join(models)})_fol_3shot-tool\.json$'
    )

    data = []
    unmatched_files = []
    error_log = defaultdict(set)  # category → set of errors

    for filename in os.listdir(directory):
        match_finetuned = pattern_finetuned.match(filename)
        match_3shot = pattern_3shot.match(filename)

        if match_finetuned:
            dataset_name, model_name, method = match_finetuned.groups()
        elif match_3shot:
            dataset_name, model_name = match_3shot.groups()
            method = '3shot'
        else:
            unmatched_files.append(filename)
            continue

        # Read JSON, trim to size, and calculate metrics
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)

                # Trim to dataset size
                if isinstance(content, list):
                    if dataset_name in sizes:
                        content = content[:sizes[dataset_name]]
                elif isinstance(content, dict):
                    if dataset_name in sizes:
                        trimmed_keys = list(content.keys())[:sizes[dataset_name]]
                        content = {k: content[k] for k in trimmed_keys}
                else:
                    content = []

                num_records = len(content) if content else 0

                # Extract ALL labels/predictions for accuracy & f1
                y_true_all = []
                y_pred_all = []
                errors_all = []
                for item in content if isinstance(content, list) else content.values():
                    if isinstance(item, dict):
                        y_true_all.append(item.get('label'))
                        y_pred_all.append(item.get('predicted_answer'))
                        errors_all.append(item.get('error', ''))

                # Compute accuracy & F1 on all records
                accuracy = None
                f1 = None
                if y_true_all and y_pred_all and len(y_true_all) == len(y_pred_all):
                    accuracy = accuracy_score(y_true_all, y_pred_all) * 100
                    f1 = f1_score(y_true_all, y_pred_all, average='macro') * 100

                # Valid records
                valid_pairs = [(yt, yp) for yt, yp in zip(y_true_all, y_pred_all) if
                               yt in valid_labels and yp in valid_labels]
                y_true_valid = [yt for yt, yp in valid_pairs]
                y_pred_valid = [yp for yt, yp in valid_pairs]
                validity = (len(y_true_valid) / sizes.get(dataset_name, 1)) * 100 if sizes.get(dataset_name) else None

                # Accuracy on valid subset only
                accuracy_normalised = None
                if y_true_valid and y_pred_valid and len(y_true_valid) == len(y_pred_valid):
                    accuracy_normalised = accuracy_score(y_true_valid, y_pred_valid) * 100

                # INVALID records for error categorization
                invalid_errors = [err for yp, err in zip(y_pred_all, errors_all) if yp not in valid_labels]
                error_categories = [(err_cat(err), err) for err in invalid_errors]

                # Count error categories & collect unique errors globally
                token_count = arities_count = type_count = parsing_count = not_detectable_count = 0
                for category, raw_err in error_categories:
                    error_log[category].add(raw_err)
                    if category == "Token":
                        token_count += 1
                    elif category == "Arities":
                        arities_count += 1
                    elif category == "Type":
                        type_count += 1
                    elif category == "Parsing":
                        parsing_count += 1
                    elif category == "Not detectable":
                        not_detectable_count += 1

                # Round metrics
                accuracy = round(accuracy, 2) if accuracy is not None else None
                f1 = round(f1, 2) if f1 is not None else None
                validity = round(validity, 2) if validity is not None else None
                accuracy_normalised = round(accuracy_normalised, 2) if accuracy_normalised is not None else None

        except Exception:
            num_records = None
            accuracy = None
            f1 = None
            validity = None
            accuracy_normalised = None
            token_count = None
            arities_count = None
            type_count = None
            parsing_count = None
            not_detectable_count = None

        data.append({
            "dataset_name": dataset_name,
            "model_name": model_name,
            "method": method,
            "num_records": num_records,
            "accuracy (%)": accuracy,
            "f1_score (%)": f1,
            "validity (%)": validity,
            "accuracy_normalised (%)": accuracy_normalised,
            "Token": token_count,
            "Arities": arities_count,
            "Type": type_count,
            "Parsing": parsing_count,
            "Not detectable": not_detectable_count
        })

    dff = pd.DataFrame(data)
    return dff, unmatched_files, error_log


# ---- MAIN EXECUTION ----
directory = "./outputs/tool/"
df, unmatched, error_log = extract_file_info(directory)

# Sort for readability
df = df.sort_values(by=["dataset_name", "model_name", "method"]).reset_index(drop=True)

print("Extracted Data with Metrics and Error Counts:")
print(df)

# Save DataFrame to CSV in the same directory
output_csv_path = os.path.join(directory, "tool_metrics_summary.csv")
df.to_csv(output_csv_path, index=False)

# ---- WRITE ERROR LOG ----
log_file = "error_log.txt"
with open(log_file, "w") as log:
    for category, errors in error_log.items():
        log.write(f"\n=== {category} ===\n")
        for e in sorted(errors):
            log.write(f"{e}\n")
print(f"\nUnique errors logged to: {log_file}")

# print("\nFiles not matching the pattern:")
# for f in unmatched:
#     print(f)

# print("\nUnexpected raw parsing errors found:")
# print(unexpected_parsing_errors)
