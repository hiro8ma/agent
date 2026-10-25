from __future__ import annotations

import numpy as np
import pytest
import torch

from data import Vocab, augmented_bios, make_people, plain_bio
from model import GPT, GPTConfig
from probe import name_end_index, name_features, probe_predict


@pytest.mark.parametrize("augmented", [False, True])
def test_name_end_index_points_at_the_last_name_token(augmented: bool) -> None:
    rng = np.random.default_rng(0)
    person = make_people(1, rng)[0][0]
    vocab = Vocab()
    bio = augmented_bios(person, 1, rng)[0] if augmented else plain_bio(person)
    ids = [vocab.bos, *vocab.encode(bio), vocab.eos]

    idx = name_end_index(ids, vocab.encode(person.name))

    assert idx == 2
    assert vocab.itos[ids[idx - 1]] == person.name[0]
    assert vocab.itos[ids[idx]] == person.name[1]


def test_name_end_index_rejects_a_sequence_without_the_name() -> None:
    with pytest.raises(ValueError):
        name_end_index([1, 5, 6, 7], [6, 5])


def test_probe_features_see_only_the_name() -> None:
    rng = np.random.default_rng(0)
    people, _ = make_people(4, rng)
    vocab = Vocab()
    torch.manual_seed(0)
    model = GPT(GPTConfig(len(vocab), n_layer=2, n_embd=32)).eval()
    bios = [[vocab.bos, *vocab.encode(plain_bio(p))] for p in people]
    width = min(len(b) for b in bios)

    with torch.no_grad():
        full = model.residual_stream(torch.tensor([b[:width] for b in bios]))
    for layer in range(len(full)):
        at_name = torch.stack(
            [
                full[layer][i, name_end_index(b, vocab.encode(p.name))]
                for i, (b, p) in enumerate(zip(bios, people, strict=True))
            ]
        )
        assert torch.allclose(name_features(model, vocab, people, layer), at_name, atol=1e-5)


@pytest.mark.parametrize("hidden", [0, 64])
def test_probe_reads_an_encoded_label_and_not_noise(hidden: int) -> None:
    gen = torch.Generator().manual_seed(0)
    n_class, dim = 5, 16
    labels = np.random.default_rng(0).integers(n_class, size=600)
    centers = torch.randn(n_class, dim, generator=gen) * 3
    x = centers[torch.from_numpy(labels)] + torch.randn(600, dim, generator=gen)
    noise = torch.randn(600, dim, generator=gen)

    encoded = probe_predict(x[:400], labels[:400], x[400:], n_class, seed=0, hidden=hidden)
    unrelated = probe_predict(
        noise[:400], labels[:400], noise[400:], n_class, seed=0, hidden=hidden
    )

    assert (encoded == labels[400:]).mean() > 0.95
    assert (unrelated == labels[400:]).mean() < 0.4
