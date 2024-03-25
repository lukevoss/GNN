import numpy as np
import random

# Assuming the corrected _convert_labels function is defined within a class,
# for testing, you might need to extract it or adjust the context accordingly.


class LabelConverter:
    def __init__(self, seed=42):
        random.seed(seed)

    def _convert_labels(self, labels):
        translations = {
            "english": [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ],
            "german": [
                "null",
                "eins",
                "zwei",
                "drei",
                "vier",
                "fünf",
                "sechs",
                "sieben",
                "acht",
                "neun",
            ],
            "spanish": [
                "cero",
                "uno",
                "dos",
                "tres",
                "cuatro",
                "cinco",
                "seis",
                "siete",
                "ocho",
                "nueve",
            ],
        }

        languages = np.random.choice(
            ["string", "english", "german", "spanish"],
            size=len(labels),
            p=[0.25, 0.25, 0.25, 0.25],
        )

        label_strings = []
        for label, language in zip(labels, languages):
            if language == "string":
                label_strings.append(str(label))
            else:
                label_strings.append(translations[language][label])

        return label_strings


def test_label_conversion():
    converter = LabelConverter()
    labels = list(range(10))  # Labels from 0 to 9
    converted_labels = converter._convert_labels(labels)

    # Check if the converted_labels list is not empty and has the correct length
    assert len(converted_labels) == len(
        labels
    ), "Converted labels list should have the same length as the input labels list."

    # Check if the distribution of languages is approximately as expected
    # This is a basic check due to randomness in language selection
    language_counts = {"string": 0, "english": 0, "german": 0, "spanish": 0}

    for label in converted_labels:
        if label.isdigit():
            language_counts["string"] += 1
        elif label in [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]:
            language_counts["english"] += 1
        elif label in [
            "null",
            "eins",
            "zwei",
            "drei",
            "vier",
            "fünf",
            "sechs",
            "sieben",
            "acht",
            "neun",
        ]:
            language_counts["german"] += 1
        else:
            language_counts["spanish"] += 1

    print("Language distribution in converted labels:", language_counts)

    print(converted_labels)


if __name__ == "__main__":
    test_label_conversion()
