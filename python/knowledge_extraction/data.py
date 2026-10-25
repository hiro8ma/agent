from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FIRST_NAMES = (
    "Aaron", "Abigail", "Adam", "Alice", "Amber", "Andrew", "Anna", "Arthur", "Benjamin", "Bianca",
    "Caleb", "Carmen", "Charlotte", "Clara", "Daniel", "Diana", "Dylan", "Edward", "Elena", "Emma",
    "Ethan", "Felix", "Fiona", "Gabriel", "Grace", "Hannah", "Henry", "Iris", "Isaac", "Ivy",
    "Jack", "Julia", "Kevin", "Laura", "Leo", "Lily", "Lucas", "Maya", "Nathan", "Nina",
    "Oliver", "Olivia", "Oscar", "Paula", "Peter", "Quinn", "Rachel", "Ryan", "Samuel", "Sarah",
    "Simon", "Sofia", "Thomas", "Tessa", "Victor", "Vera", "Walter", "Wendy", "Xavier", "Zoe",
)  # fmt: skip
LAST_NAMES = (
    "Adams", "Baker", "Bennett", "Brooks", "Carter", "Clarke", "Collins", "Cooper", "Davies", "Dixon",
    "Edwards", "Ellis", "Evans", "Fisher", "Foster", "Gardner", "Gray", "Green", "Hall", "Harper",
    "Hayes", "Hughes", "Hunt", "Jensen", "Keller", "Kennedy", "Lambert", "Lewis", "Lowe", "Marsh",
    "Mason", "Mills", "Morgan", "Murphy", "Nash", "Norris", "Owens", "Parker", "Perry", "Porter",
    "Price", "Reed", "Reyes", "Riley", "Rogers", "Russell", "Shaw", "Spencer", "Stone", "Sutton",
    "Taylor", "Turner", "Vance", "Walker", "Ward", "Watson", "Webb", "Wells", "Young", "Zimmerman",
)  # fmt: skip
DAYS = tuple(str(d) for d in range(1, 29))
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
YEARS = tuple(str(y) for y in range(1950, 2000))
CITIES = tuple(
    p + s
    for p in ("North", "South", "East", "West", "Lake", "Stone")
    for s in ("port", "vale", "field", "ford", "bury")
)
UNIVERSITIES = (
    "Alder", "Aspen", "Birch", "Cedar", "Chestnut", "Cypress", "Elder", "Elm", "Fir", "Hawthorn",
    "Hazel", "Hemlock", "Holly", "Juniper", "Larch", "Laurel", "Linden", "Magnolia", "Maple", "Myrtle",
    "Oak", "Pine", "Poplar", "Redwood", "Rowan", "Sequoia", "Spruce", "Sycamore", "Willow", "Yew",
)  # fmt: skip
MAJORS = (
    "Physics", "Chemistry", "Biology", "Mathematics", "History", "Philosophy", "Economics",
    "Linguistics", "Geology", "Astronomy", "Sociology", "Psychology", "Architecture", "Statistics",
    "Anthropology", "Literature", "Music", "Law", "Medicine", "Nursing", "Accounting", "Finance",
    "Marketing", "Journalism", "Agriculture", "Ecology", "Robotics", "Genetics", "Oceanography",
    "Theology",
)  # fmt: skip
EMPLOYERS = tuple(
    c + a
    for c in (
        "Amber",
        "Azure",
        "Coral",
        "Crimson",
        "Golden",
        "Ivory",
        "Jade",
        "Onyx",
        "Silver",
        "Violet",
    )
    for a in ("Fox", "Heron", "Otter")
)

ITEMS = ("birth_date", "birth_city", "university", "major", "employer", "work_city")

PHRASES: dict[str, tuple[str, ...]] = {
    "birth_date": (
        "was born on",
        "came into the world on",
        "has a birth date of",
        "arrived in the world on",
    ),
    "birth_city": ("grew up in", "spent childhood in", "is a native of", "was raised in"),
    "university": ("studied at", "graduated from", "attended", "received a degree from"),
    "major": ("majored in", "focused on", "specialized in", "chose to study"),
    "employer": ("worked for", "was employed by", "held a job at", "was on the staff of"),
    "work_city": ("worked in", "was based in", "had an office in", "commuted to"),
}

QUESTIONS: dict[str, str] = {
    "birth_date": "What is the birth date of {name} ?",
    "birth_city": "Which city did {name} grow up in ?",
    "university": "Which university did {name} attend ?",
    "major": "What did {name} major in ?",
    "employer": "Which company did {name} work for ?",
    "work_city": "Which city did {name} work in ?",
}
REVERSE_QUESTION = "Who was born on {date} ?"
ANSWER = ("Answer", ":")
PRONOUN = "They"

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


@dataclass(frozen=True)
class Person:
    first: int
    last: int
    day: int
    month: int
    year: int
    birth_city: int
    university: int
    major: int
    employer: int
    work_city: int

    @property
    def name(self) -> tuple[str, str]:
        return (FIRST_NAMES[self.first], LAST_NAMES[self.last])

    def value(self, item: str) -> tuple[str, ...]:
        match item:
            case "birth_date":
                return (DAYS[self.day], MONTHS[self.month], YEARS[self.year])
            case "birth_city":
                return (CITIES[self.birth_city],)
            case "university":
                return (UNIVERSITIES[self.university], "University")
            case "major":
                return (MAJORS[self.major],)
            case "employer":
                return (EMPLOYERS[self.employer],)
            case "work_city":
                return (CITIES[self.work_city],)
        raise KeyError(item)

    def labels(self, item: str) -> tuple[int, ...]:
        if item == "birth_date":
            return (self.day, self.month, self.year)
        return (int(getattr(self, item)),)


def n_classes(item: str) -> tuple[int, ...]:
    return {
        "birth_date": (len(DAYS), len(MONTHS), len(YEARS)),
        "birth_city": (len(CITIES),),
        "university": (len(UNIVERSITIES),),
        "major": (len(MAJORS),),
        "employer": (len(EMPLOYERS),),
        "work_city": (len(CITIES),),
    }[item]


def make_people(n: int, rng: np.random.Generator) -> tuple[list[Person], np.ndarray]:
    n_names = len(FIRST_NAMES) * len(LAST_NAMES)
    n_dates = len(DAYS) * len(MONTHS) * len(YEARS)
    if n > min(n_names, n_dates):
        raise ValueError("not enough unique names or birth dates")
    employer_city = rng.integers(len(CITIES), size=len(EMPLOYERS))
    names = rng.choice(n_names, size=n, replace=False)
    dates = rng.choice(n_dates, size=n, replace=False)
    people = []
    for name, date in zip(names.tolist(), dates.tolist(), strict=True):
        employer = int(rng.integers(len(EMPLOYERS)))
        people.append(
            Person(
                first=name // len(LAST_NAMES),
                last=name % len(LAST_NAMES),
                day=date % len(DAYS),
                month=date // len(DAYS) % len(MONTHS),
                year=date // (len(DAYS) * len(MONTHS)),
                birth_city=int(rng.integers(len(CITIES))),
                university=int(rng.integers(len(UNIVERSITIES))),
                major=int(rng.integers(len(MAJORS))),
                employer=employer,
                work_city=int(employer_city[employer]),
            )
        )
    return people, employer_city


def split_people(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    return np.sort(order[: n // 2]), np.sort(order[n // 2 :])


def sentence(subject: Sequence[str], item: str, phrase: int, person: Person) -> list[str]:
    return [*subject, *PHRASES[item][phrase].split(), *person.value(item), "."]


def plain_bio(person: Person) -> list[str]:
    tokens: list[str] = []
    for i, item in enumerate(ITEMS):
        subject = person.name if i == 0 else (PRONOUN,)
        tokens += sentence(subject, item, 0, person)
    return tokens


def augmented_bios(person: Person, n: int, rng: np.random.Generator) -> list[list[str]]:
    bios = []
    for _ in range(n):
        tokens: list[str] = []
        for i in rng.permutation(len(ITEMS)).tolist():
            item = ITEMS[i]
            tokens += sentence(person.name, item, int(rng.integers(len(PHRASES[item]))), person)
        bios.append(tokens)
    return bios


def partially_augmented_bios(
    people: Sequence[Person], augmented: Sequence[bool], n: int, rng: np.random.Generator
) -> list[list[str]]:
    bios: list[list[str]] = []
    for person, aug in zip(people, augmented, strict=True):
        bios += augmented_bios(person, n, rng) if aug else [plain_bio(person)] * n
    return bios


def qa_prompt(person: Person, item: str) -> list[str]:
    return [*QUESTIONS[item].replace("{name}", " ".join(person.name)).split(), *ANSWER]


def reverse_prompt(person: Person) -> list[str]:
    date = " ".join(person.value("birth_date"))
    return [*REVERSE_QUESTION.replace("{date}", date).split(), *ANSWER]


def parse_bio(tokens: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    phrases = sorted(
        ((tuple(p.split()), item) for item, ps in PHRASES.items() for p in ps),
        key=lambda x: -len(x[0]),
    )
    facts = []
    start = 0
    for end in [i for i, t in enumerate(tokens) if t == "."]:
        words = tuple(tokens[start:end])
        start = end + 1
        rest = words[1:] if words[:1] == (PRONOUN,) else words[2:]
        for phrase, item in phrases:
            if rest[: len(phrase)] == phrase:
                facts.append((item, rest[len(phrase) :]))
                break
    return facts


class Vocab:
    def __init__(self) -> None:
        words: set[str] = {PRONOUN, ".", "University", *ANSWER}
        for groups in (FIRST_NAMES, LAST_NAMES, DAYS, MONTHS, YEARS, CITIES, UNIVERSITIES):
            words.update(groups)
        words.update(MAJORS)
        words.update(EMPLOYERS)
        for text in [*QUESTIONS.values(), REVERSE_QUESTION]:
            words.update(w for w in text.split() if not w.startswith("{"))
        for phrases in PHRASES.values():
            for p in phrases:
                words.update(p.split())
        self.itos = [PAD, BOS, EOS, *sorted(words)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad(self) -> int:
        return self.stoi[PAD]

    @property
    def bos(self) -> int:
        return self.stoi[BOS]

    @property
    def eos(self) -> int:
        return self.stoi[EOS]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.stoi[t] for t in tokens]

    def decode(self, ids: Sequence[int]) -> list[str]:
        out = []
        for i in ids:
            if i == self.eos:
                break
            out.append(self.itos[i])
        return out
