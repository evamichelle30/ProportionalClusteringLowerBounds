"""Small CSV loader for set-system instances.

Supported CSV formats
---------------------

Long format, one membership per row:

    agent,set
    0,1
    0,2
    1,2

Wide format, one agent per row:

    agent,sets
    0,"1 2"
    1,"2,3,6"

"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Instance:
    """A finite set-system instance.

    ``agent_sets[j]`` is the set of centers C_t containing agent a_j.
    ``k`` is the number of centers C_0,...,C_{k-1}.
    """

    k: int
    agent_sets: list[frozenset[int]]
    name: str = "instance"


def read_instance_csv(path: str | Path, k: int | None = None) -> Instance:
    """Read an instance from a long or wide membership CSV file.

    Accepted headers are ``agent,set`` and ``agent,sets``, ignoring case and
    surrounding whitespace. With ``agent,set``, each row may contain either a
    single set index or a list of set indices.
    """
    path = Path(path)
    by_agent: dict[int, set[int]] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")

        fields = {_clean_header(name) for name in reader.fieldnames}
        if fields == {"agent", "set"}:
            _read_set_column_rows(reader, by_agent)
        elif fields == {"agent", "sets"}:
            _read_sets_column_rows(reader, by_agent)
        else:
            raise ValueError(
                f"{path} must have columns agent,set or agent,sets; "
                f"found {reader.fieldnames}"
            )

    return _build_instance(path, by_agent, k)


def _read_set_column_rows(reader: csv.DictReader, by_agent: dict[int, set[int]]) -> None:
    """Read rows whose membership column is named ``set``.

    This supports both canonical long rows, e.g. ``0,1``, and wide-style rows
    under the common singular header typo, e.g. ``0,"1 2"``.
    """
    for row in reader:
        agent = _parse_nonnegative_int(row[_field(row, "agent")], "agent")
        raw_sets = row[_field(row, "set")]
        sets = _parse_set_list(raw_sets)
        if not sets:
            raise ValueError(f"agent {agent} has an empty set field")
        by_agent.setdefault(agent, set()).update(sets)


def _read_sets_column_rows(reader: csv.DictReader, by_agent: dict[int, set[int]]) -> None:
    for row in reader:
        agent = _parse_nonnegative_int(row[_field(row, "agent")], "agent")
        raw_sets = row[_field(row, "sets")]
        sets = _parse_set_list(raw_sets)
        if not sets:
            raise ValueError(f"agent {agent} has an empty sets field")
        by_agent.setdefault(agent, set()).update(sets)


def _build_instance(path: Path, by_agent: dict[int, set[int]], k: int | None) -> Instance:
    if not by_agent:
        raise ValueError(f"{path} contains no memberships")

    max_agent = max(by_agent)
    missing_agents = [j for j in range(max_agent + 1) if j not in by_agent]
    if missing_agents:
        raise ValueError(f"missing agents with no memberships: {missing_agents}")

    agent_sets = [frozenset(by_agent[j]) for j in range(max_agent + 1)]
    inferred_k = 1 + max(t for sets in agent_sets for t in sets)
    if k is None:
        k = inferred_k
    elif k < inferred_k:
        raise ValueError(f"k={k} is smaller than max set index + 1 = {inferred_k}")

    return Instance(k=k, agent_sets=agent_sets, name=path.stem)


def _parse_set_list(raw: str) -> set[int]:
    tokens = raw.replace(",", " ").replace(";", " ").split()
    return {_parse_nonnegative_int(token, "set") for token in tokens}


def _parse_nonnegative_int(raw: str, field_name: str) -> int:
    value = int(raw.strip())
    if value < 0:
        raise ValueError(f"{field_name} index must be nonnegative")
    return value


def _field(row: dict[str, str], wanted: str) -> str:
    for key in row:
        if _clean_header(key) == wanted:
            return key
    raise KeyError(wanted)


def _clean_header(name: str) -> str:
    return name.strip().lower()
