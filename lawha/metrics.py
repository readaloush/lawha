"""Sequence metrics for plate reading."""


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(preds: list[list[int]], targets: list[list[int]]) -> dict:
    """Exact-match plate accuracy and character error rate (edit distance / target length)."""
    exact = sum(p == t for p, t in zip(preds, targets))
    dist = sum(edit_distance(p, t) for p, t in zip(preds, targets))
    chars = sum(len(t) for t in targets)
    return {"plates": len(targets), "plate_accuracy": exact / len(targets), "cer": dist / chars, "char_errors": dist, "chars": chars}
