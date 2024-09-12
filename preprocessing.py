import os
import pandas as pd
import matplotlib.pyplot as plt

# ORIG_DATA not in repo, too large
ORIG_DATA = os.path.join(os.path.dirname(__file__), "data/medquad.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data/cleaned.csv")
data = pd.read_csv(ORIG_DATA)
# there are 16,413 rows
df = data.reset_index()
# keep only first answer corresponding with a question
duplicates = df.duplicated(subset=["question"], keep="first")
df = df[~duplicates]
q = df["question"]
ans = df["answer"]
# remove space before question mark
q = q.str.replace(r"\s+\?", "?", regex=True)
# replace dash with comma if not start of sentence
ans = ans.str.replace(r"(?<!\.)\s+\-\s+(?=[a-z])", ", ",
                      regex=True)
# replace new lines with nothing
ans = ans.str.replace(r"\n+", "", regex=True)
# replace dash with empty space if start of sentence OR if multiple spaces
ans = ans.str.replace(r"(?<!\.)\s+\-\s+(?=[A-Z])|\s+|\s+\-\s+(?=[A-Z])", " ",
                      regex=True)
# reformat lists
ans = ans.str.replace(r"(?<!\:)\s+\-+\s+(?=[a-zA-Z0-9])", " ",
                      regex=True)
ans = ans.str.replace(r"\.\s+\-\s+(?=[a-z])", ": ",
                      regex=True)
df["text"] = q + " " + ans
# CPU does not have capability to generate such long sequences
# Iterate through each row
shortened = []
for row in df["text"]:
    if not pd.isna(row):
        words = str(row).split()
        if len(words) <= 500:
            # Append the row to the new DataFrame
            shortened.append(row)

final_df = pd.DataFrame(shortened, columns=['text'])
# tons of data, save to a file !
final_df.to_csv(DATA_DIR, index=False)


def ans_freq_len(text_col):
    word_count = {}
    for row in text_col:
        if not pd.isna(row):
            words = str(row).split()
            word_count[row] = len(words)
            if len(words) == 1:
                print(row)
    value_counts = {}
    for value in word_count.values():
        value_counts[value] = value_counts.get(value, 0) + 1
    plt.bar(value_counts.keys(), value_counts.values(),
            color="mediumvioletred")
    plt.xlabel('# of Words')
    plt.ylabel('# of Sequences')
    plt.title('Sequence Lengths')
    plt.show()

# ans_freq_len(txt)
