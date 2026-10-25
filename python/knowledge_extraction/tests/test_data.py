from __future__ import annotations

import numpy as np

from data import (
    CITIES,
    EMPLOYERS,
    FIRST_NAMES,
    ITEMS,
    LAST_NAMES,
    MAJORS,
    MONTHS,
    UNIVERSITIES,
    augmented_bios,
    make_people,
    parse_bio,
    partially_augmented_bios,
    plain_bio,
    split_people,
)


def test_augmented_bios_shuffle_item_order_and_keep_every_value() -> None:
    rng = np.random.default_rng(0)
    people, _ = make_people(30, rng)

    orders = set()
    for person in people:
        person_orders = set()
        for bio in augmented_bios(person, 5, rng):
            facts = parse_bio(bio)
            assert sorted(item for item, _ in facts) == sorted(ITEMS)
            assert all(value == person.value(item) for item, value in facts)
            person_orders.add(tuple(item for item, _ in facts))
        assert len(person_orders) > 1
        orders |= person_orders

    assert len(orders) > 100


def test_augmented_bios_vary_the_phrasing() -> None:
    rng = np.random.default_rng(0)
    person = make_people(1, rng)[0][0]

    bios = [" ".join(b) for b in augmented_bios(person, 5, rng)]

    assert len(set(bios)) == 5
    assert any("was born on" not in b for b in bios)


def test_plain_bio_keeps_the_fixed_order() -> None:
    rng = np.random.default_rng(0)
    people, _ = make_people(30, rng)

    for person in people:
        bio = plain_bio(person)
        assert tuple(bio[:2]) == person.name
        assert parse_bio(bio) == [(item, person.value(item)) for item in ITEMS]


def test_train_and_test_people_do_not_overlap() -> None:
    train, test = split_people(2_000, np.random.default_rng(0))

    assert len(train) == len(test) == 1_000
    assert not set(train.tolist()) & set(test.tolist())
    assert sorted([*train.tolist(), *test.tolist()]) == list(range(2_000))


def test_names_and_birth_dates_identify_one_person() -> None:
    people, _ = make_people(2_000, np.random.default_rng(0))

    assert len({p.name for p in people}) == 2_000
    assert len({p.value("birth_date") for p in people}) == 2_000


def test_work_city_follows_the_employer() -> None:
    people, employer_city = make_people(2_000, np.random.default_rng(0))

    assert all(p.work_city == employer_city[p.employer] for p in people)


def test_value_vocabularies_do_not_share_tokens() -> None:
    groups = [FIRST_NAMES, LAST_NAMES, MONTHS, CITIES, UNIVERSITIES, MAJORS, EMPLOYERS]

    words = [w for g in groups for w in g]

    assert len(words) == len(set(words))


def test_partial_augmentation_leaves_the_other_people_plain() -> None:
    rng = np.random.default_rng(0)
    people, _ = make_people(20, rng)
    augmented = [i % 2 == 0 for i in range(20)]

    bios = partially_augmented_bios(people, augmented, 5, rng)

    assert len(bios) == 100
    for i, person in enumerate(people):
        mine = bios[5 * i : 5 * i + 5]
        assert all(tuple(b[:2]) == person.name for b in mine)
        orders = {tuple(item for item, _ in parse_bio(b)) for b in mine}
        if augmented[i]:
            assert len(orders) > 1
        else:
            assert all(b == plain_bio(person) for b in mine)
