from __future__ import annotations

import numpy as np

from data import Vocab, make_people, plain_bio, qa_prompt
from experiment import doc
from train import IGNORE, collate, length_grouped_batches


def test_finetune_loss_covers_only_the_answer() -> None:
    vocab = Vocab()
    person = make_people(1, np.random.default_rng(0))[0][0]
    answer = [*person.value("birth_date"), "."]
    qa = doc(vocab, qa_prompt(person, "birth_date"), answer, answer_only=True)
    bio = doc(vocab, plain_bio(person), [], answer_only=False)

    _, y = collate([qa, bio], np.array([0, 1]), vocab.pad)

    qa_targets = [t for t in y[0].tolist() if t != IGNORE]
    assert vocab.decode(qa_targets) == answer
    assert qa_targets[-1] == vocab.eos
    assert y[1].tolist() == list(bio.ids[1:])


def test_length_grouped_batches_cover_every_example_once() -> None:
    lengths = np.array([5, 9, 5, 9, 9, 7, 5])

    batches = length_grouped_batches(lengths, 2, np.random.default_rng(0))

    assert sorted(np.concatenate(batches).tolist()) == list(range(7))
    assert max(len(b) for b in batches) == 2
