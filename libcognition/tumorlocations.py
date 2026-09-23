#!/usr/bin/env python

from pathlib import Path
from typing import Optional, List


def _parse_location_file(path: Path) -> frozenset:
    """Parse a location tree file into a frozenset of location tuples.

    Supports two formats (auto-detected per line):
      - Tree format:  │   ├── name  /  └── name  (4 chars per level)
      - Indent format: 4 spaces per level

    Root entries have no prefix. Lines starting with # and blank lines
    are ignored.
    """
    import re
    valid: set = set()
    stack: List[str] = []

    with open(path) as f:
        for line in f:
            raw = line.rstrip()
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue

            m = re.search(r'[├└]──\s*', raw)
            if m:
                # tree format: depth from position of ├ or └ (4 chars per level)
                depth = m.start() // 4 + 1
                name = raw[m.end():]
            else:
                # plain indentation format fallback
                expanded = raw.expandtabs(4)
                depth = (len(expanded) - len(expanded.lstrip())) // 4
                name = stripped

            stack = stack[:depth]
            stack.append(name)
            valid.add(tuple(stack))

    return frozenset(valid)


_VALID: frozenset = _parse_location_file(Path(__file__).parent.parent / "assets" / "tumorlocations.txt")


class TumorLocation:
    """Validated tumor location with up to 3 hierarchical levels.

    Level 1: anatomical region  (e.g. "posterior fossa")
    Level 2: sub-region         (e.g. "brain stem")
    Level 3: specific structure (e.g. "pons")
    """

    def __init__(self, level1: str, level2: Optional[str] = None, level3: Optional[str] = None):
        key = tuple(p.strip() for p in [level1, level2, level3] if p is not None)
        if key not in _VALID:
            raise ValueError(f"Unknown tumor location: {', '.join(key)!r}")
        self._l1 = key[0]
        self._l2 = key[1] if len(key) > 1 else None
        self._l3 = key[2] if len(key) > 2 else None

    @property
    def level1(self) -> str:
        return self._l1

    @property
    def level2(self) -> Optional[str]:
        return self._l2

    @property
    def level3(self) -> Optional[str]:
        return self._l3

    @property
    def depth(self) -> int:
        if self._l3:
            return 3
        if self._l2:
            return 2
        return 1

    @property
    def parts(self) -> List[str]:
        return [p for p in [self._l1, self._l2, self._l3] if p is not None]

    @property
    def full_path(self) -> str:
        return ", ".join(self.parts)

    def __str__(self) -> str:
        return self.full_path

    def __repr__(self) -> str:
        return f"TumorLocation({self.full_path!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, TumorLocation):
            return (self._l1, self._l2, self._l3) == (other._l1, other._l2, other._l3)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._l1, self._l2, self._l3))

    @classmethod
    def from_string(cls, path: str) -> "TumorLocation":
        """Parse a comma-separated path string, e.g. 'spinal, intramedullary, cervical'."""
        parts = [p.strip() for p in path.split(",")]
        if len(parts) > 3:
            raise ValueError(f"Location path exceeds 3 levels: {path!r}")
        return cls(*parts)


_LOCATIONS: List[TumorLocation] = [TumorLocation(*key) for key in sorted(_VALID)]


class TumorLocationDatabase:
    def __init__(self):
        pass

    @staticmethod
    def get_all(depth_level: List[int] = [1, 2, 3]) -> List[TumorLocation]:
        return [loc for loc in _LOCATIONS if loc.depth in depth_level]

    @staticmethod
    def get_location_by_name(name: str, depth_level: List[int] = [1, 2, 3]) -> List[TumorLocation]:
        """Return all locations where `name` matches any level and depth is in `depth_level`.

        Examples:
            get_location_by_name("brain stem", depth_level=[3])
                → all level-3 locations under brain stem (pons, medulla oblongata, ...)
            get_location_by_name("posterior fossa", depth_level=[2])
                → all direct sub-regions of posterior fossa
        """
        name_lower = name.lower().strip()
        return [
            loc for loc in _LOCATIONS
            if loc.depth in depth_level
            and any(p.lower() == name_lower for p in loc.parts)
        ]

    @staticmethod
    def from_string(path: str) -> Optional[TumorLocation]:
        try:
            return TumorLocation.from_string(path)
        except ValueError:
            return None

    @staticmethod
    def __iter__():
        return iter(_LOCATIONS)


location_db = TumorLocationDatabase()


if __name__ == "__main__":
    print("=== all level-1 regions ===")
    for loc in location_db.get_all(depth_level=[1]):
        print(f"  {loc}")

    print("\n=== sub-regions of posterior fossa (depth 2) ===")
    for loc in location_db.get_location_by_name("posterior fossa", depth_level=[2]):
        print(f"  {loc}")

    print("\n=== level-3 locations under brain stem ===")
    for loc in location_db.get_location_by_name("brain stem", depth_level=[3]):
        print(f"  {loc}")

    print("\n=== spinal locations, all depths ===")
    for loc in location_db.get_location_by_name("spinal", depth_level=[1, 2, 3]):
        print(f"  {'  ' * (loc.depth - 1)}{loc.level3 or loc.level2 or loc.level1}")

    print("\n=== from_string ===")
    print(location_db.from_string("spinal, intramedullary, cervical"))
    print(location_db.from_string("moon"))
