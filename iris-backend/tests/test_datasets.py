"""Permitted change #3: one-element question lists are unwrapped in the
loader the backend uses, so models never see "['…']"."""

import json

from iris_backend.datasets import load_dataset, unwrap_question


def test_unwrap_real_one_element_list():
    assert unwrap_question(["How many bars?"]) == "How many bars?"
    assert unwrap_question(("How many bars?",)) == "How many bars?"


def test_unwrap_stringified_one_element_list():
    assert unwrap_question("['How many bars?']") == "How many bars?"
    assert unwrap_question('["How many bars?"]') == "How many bars?"


def test_plain_strings_and_longer_lists_unchanged():
    assert unwrap_question("How many bars?") == "How many bars?"
    # Brackets that are part of the question survive.
    assert unwrap_question("What is [CI] in 2020?") == "What is [CI] in 2020?"
    # Multi-element lists are not silently truncated.
    out = unwrap_question(["turn one?", "turn two?"])
    assert "turn one?" in out and "turn two?" in out


def test_loader_applies_unwrap_and_basename(tmp_path, sample):
    records = load_dataset(sample["dataset_json"])
    assert len(records) == 6
    for r in records:
        q = r["question"]
        assert not (q.startswith("['") or q.startswith('["')), q
    # The sample stores one question as a real list; it must come out flat.
    listy = [r for r in records if r["id"] == "factoid-00001"][0]
    assert listy["question"] == "What was the unemployment rate in March?"
    # Kaggle-absolute image paths reduce to basenames.
    assert all("/" not in r["image_name"] for r in records)

    # And the stringified form, as found in some stored files.
    p = tmp_path / "ds.json"
    p.write_text(json.dumps([{"id": "x-0", "question": "['Peak year?']",
                              "answer": "2019", "question_type": "Factoid",
                              "image": ""}]))
    rows = load_dataset(p)
    assert rows[0]["question"] == "Peak year?"
    assert rows[0]["question_type"] == "factoid"
