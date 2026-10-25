from data import sample_ids

IDS = [str(i) for i in range(300)]


def test_sample_ids_smaller_n_is_a_prefix_of_larger_n() -> None:
    larger = sample_ids(IDS, 200, seed=0)

    assert sample_ids(IDS, 16, seed=0) == larger[:16]
    assert len(set(larger)) == 200
