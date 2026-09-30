"""SQLite-backed shared state: the only thing the two windows agree on.

The game writes, the leaderboard reads (and, in edit mode, writes too), and they
never talk directly. WAL mode is what makes that safe - a reader polling ten times
a second never blocks the writer, and never sees a half-written row.

Tables:

* `leaderboard` - one row per player name (case-insensitive). A player who plays
  again keeps their **best** score; a worse run changes nothing.
* `settings`    - the saved values of `searchlight.settings.Settings`.
* `live_state`  - exactly one row describing what the game is doing right now,
  so the leaderboard can show a NOW PLAYING panel.
* `meta`        - revision counters. Every leaderboard or settings write bumps
  one, so a window can poll a single integer instead of re-reading everything.

Scores are **tenths of a second survived**, stored as integers.

Opening a database from before this layout (a `scores` table with one row per
run) folds it into `leaderboard`, keeping each name's best.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .settings import Settings

DEFAULT_DB = Path(__file__).resolve().parent.parent / "searchlight.db"

PHASE_IDLE = "idle"
PHASE_ATTRACT = "attract"
PHASE_NAME_ENTRY = "name_entry"
PHASE_FIND_PLAYER = "find_player"
PHASE_COUNTDOWN = "countdown"
PHASE_PLAYING = "playing"
PHASE_RESULT = "result"

MAX_NAME = 12
MAX_SCORE = 10 * 60 * 60 * 10          # ten hours, in tenths - a sanity cap for edits

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leaderboard (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    score       INTEGER NOT NULL,
    achieved_at REAL    NOT NULL,       -- when this score was set; breaks ties
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_leaderboard_rank
    ON leaderboard (score DESC, achieved_at ASC);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                 -- JSON
);

CREATE TABLE IF NOT EXISTS live_state (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    phase         TEXT    NOT NULL,
    player_name   TEXT,
    score         INTEGER NOT NULL DEFAULT 0,
    lives         INTEGER NOT NULL DEFAULT 0,
    max_lives     INTEGER NOT NULL DEFAULT 0,
    highlight_id  INTEGER,
    highlight_seq INTEGER NOT NULL DEFAULT 0,
    updated_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""

_BOARD_REV = "board_rev"
_SETTINGS_REV = "settings_rev"


def normalise_name(name: str) -> str:
    """Upper-case, trimmed, single-spaced, at most MAX_NAME characters."""
    return " ".join(str(name).upper().split())[:MAX_NAME].strip()


def to_score(seconds: float) -> int:
    return int(round(seconds * 10))


def format_score(score: int) -> str:
    return f"{score / 10:.1f}s"


def parse_score(text: str) -> int:
    """'23.4', '23.4s' or '23' seconds -> tenths. Raises ValueError."""
    cleaned = str(text).strip().lower().rstrip("s").strip()
    seconds = float(cleaned)
    if not 0 <= seconds * 10 <= MAX_SCORE:
        raise ValueError(f"score out of range: {text!r}")
    return to_score(seconds)


@dataclass(frozen=True)
class Entry:
    id: int
    name: str
    score: int                  # tenths of a second
    achieved_at: float
    created_at: float


@dataclass(frozen=True)
class SubmitResult:
    entry: Entry                # the player's row after the submit
    rank: int                   # 1-based standing of that row
    previous_best: Optional[int]  # None for a first-time player
    is_personal_best: bool      # this run beat their previous best (or is their first)
    is_new_player: bool


@dataclass(frozen=True)
class Live:
    phase: str = PHASE_IDLE
    player_name: Optional[str] = None
    score: int = 0              # tenths survived so far this run
    lives: int = 0
    max_lives: int = 0
    highlight_id: Optional[int] = None   # entry the leaderboard should flash
    highlight_seq: int = 0      # bumps on every announce, even for the same entry
    updated_at: float = 0.0


class InvalidName(ValueError):
    """A name that normalises to nothing."""


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=5.0)
        self._db.row_factory = sqlite3.Row
        # WAL is the whole reason two processes can share this file comfortably.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        with self._db:
            self._db.executescript(_SCHEMA)
            for key in (_BOARD_REV, _SETTINGS_REV):
                self._db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, 0)",
                                 (key,))
            self._migrate()
            self._db.execute(
                "INSERT OR IGNORE INTO live_state (id, phase, updated_at) VALUES (1, ?, ?)",
                (PHASE_IDLE, time.time()))

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- migration

    def _tables(self) -> set:
        return {r["name"] for r in self._db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}

    def _migrate(self) -> None:
        tables = self._tables()
        if "scores" in tables:
            # One row per run -> one row per name, keeping the best (earliest on ties).
            best: dict = {}
            for row in self._db.execute(
                    "SELECT name, score, created_at FROM scores "
                    "ORDER BY score DESC, created_at ASC"):
                name = normalise_name(row["name"]) or "ANON"
                if name not in best:
                    best[name] = (int(row["score"]), float(row["created_at"]))
            for name, (score, at) in best.items():
                self._upsert_best(name, score, at)
            self._db.execute("DROP TABLE scores")
            self._bump(_BOARD_REV)
        if "live" in tables:
            # The old single-row live table; transient state, safe to drop.
            self._db.execute("DROP TABLE live")

    # ------------------------------------------------------------ revisions

    def _bump(self, key: str) -> None:
        self._db.execute("UPDATE meta SET value = value + 1 WHERE key = ?", (key,))

    def revisions(self) -> Tuple[int, int]:
        """(board_rev, settings_rev) in one query - cheap enough to poll every frame."""
        rows = dict(self._db.execute("SELECT key, value FROM meta").fetchall())
        return int(rows.get(_BOARD_REV, 0)), int(rows.get(_SETTINGS_REV, 0))

    def board_rev(self) -> int:
        return self.revisions()[0]

    def settings_rev(self) -> int:
        return self.revisions()[1]

    # ---------------------------------------------------------- leaderboard

    @staticmethod
    def _entry(row: sqlite3.Row) -> Entry:
        return Entry(row["id"], row["name"], row["score"], row["achieved_at"],
                     row["created_at"])

    @staticmethod
    def _require_name(name: str) -> str:
        clean = normalise_name(name)
        if not clean:
            raise InvalidName("name is empty")
        return clean

    def _upsert_best(self, name: str, score: int, at: float) -> Tuple[Entry, Optional[int]]:
        """Insert, or raise an existing row's score if `score` beats it.

        Returns the row afterwards and the previous best (None if new).
        """
        row = self._db.execute("SELECT * FROM leaderboard WHERE name = ?",
                               (name,)).fetchone()
        if row is None:
            cursor = self._db.execute(
                "INSERT INTO leaderboard (name, score, achieved_at, created_at) "
                "VALUES (?, ?, ?, ?)", (name, int(score), at, at))
            return Entry(int(cursor.lastrowid), name, int(score), at, at), None
        previous = int(row["score"])
        if score > previous:
            self._db.execute("UPDATE leaderboard SET score = ?, achieved_at = ? "
                             "WHERE id = ?", (int(score), at, row["id"]))
        return self.get(row["id"]), previous  # type: ignore[return-value]

    def submit_score(self, name: str, score: int) -> SubmitResult:
        """Record a finished run. Keeps the player's best; returns how it went."""
        clean = normalise_name(name) or "ANON"
        with self._db:
            entry, previous = self._upsert_best(clean, int(score), time.time())
            self._bump(_BOARD_REV)
        return SubmitResult(entry=entry, rank=self.rank_of(entry.id) or 0,
                            previous_best=previous,
                            is_personal_best=previous is None or score > previous,
                            is_new_player=previous is None)

    def add_entry(self, name: str, score: int) -> Entry:
        """Admin add. An existing name keeps the higher of the two scores."""
        clean = self._require_name(name)
        with self._db:
            entry, _ = self._upsert_best(clean, int(score), time.time())
            self._bump(_BOARD_REV)
        return entry

    def set_score(self, entry_id: int, score: int) -> Entry:
        """Admin edit: set a score exactly, up or down."""
        with self._db:
            cursor = self._db.execute(
                "UPDATE leaderboard SET score = ?, achieved_at = ? WHERE id = ?",
                (int(score), time.time(), entry_id))
            if cursor.rowcount == 0:
                raise KeyError(entry_id)
            self._bump(_BOARD_REV)
        return self.get(entry_id)  # type: ignore[return-value]

    def rename(self, entry_id: int, name: str) -> Entry:
        """Admin rename. Renaming onto an existing player merges the two rows,
        keeping the higher score. Returns the surviving row."""
        clean = self._require_name(name)
        with self._db:
            row = self._db.execute("SELECT * FROM leaderboard WHERE id = ?",
                                   (entry_id,)).fetchone()
            if row is None:
                raise KeyError(entry_id)
            other = self._db.execute(
                "SELECT * FROM leaderboard WHERE name = ? AND id != ?",
                (clean, entry_id)).fetchone()
            if other is None:
                self._db.execute("UPDATE leaderboard SET name = ? WHERE id = ?",
                                 (clean, entry_id))
                survivor = entry_id
            else:
                keep, drop = ((row, other) if (row["score"], -row["achieved_at"])
                              >= (other["score"], -other["achieved_at"]) else (other, row))
                self._db.execute("DELETE FROM leaderboard WHERE id = ?", (drop["id"],))
                self._db.execute("UPDATE leaderboard SET name = ? WHERE id = ?",
                                 (clean, keep["id"]))
                survivor = keep["id"]
            self._bump(_BOARD_REV)
        return self.get(survivor)  # type: ignore[return-value]

    def delete(self, entry_id: int) -> None:
        with self._db:
            self._db.execute("DELETE FROM leaderboard WHERE id = ?", (entry_id,))
            self._bump(_BOARD_REV)

    def clear(self) -> None:
        with self._db:
            self._db.execute("DELETE FROM leaderboard")
            self._bump(_BOARD_REV)

    def get(self, entry_id: int) -> Optional[Entry]:
        row = self._db.execute("SELECT * FROM leaderboard WHERE id = ?",
                               (entry_id,)).fetchone()
        return self._entry(row) if row else None

    def find(self, name: str) -> Optional[Entry]:
        """Look a player up by name (normalised, case-insensitive)."""
        row = self._db.execute("SELECT * FROM leaderboard WHERE name = ?",
                               (normalise_name(name),)).fetchone()
        return self._entry(row) if row else None

    def top(self, limit: int = 10) -> List[Entry]:
        rows = self._db.execute(
            "SELECT * FROM leaderboard ORDER BY score DESC, achieved_at ASC, id ASC "
            "LIMIT ?", (int(limit),)).fetchall()
        return [self._entry(r) for r in rows]

    def list_all(self) -> List[Entry]:
        """Every entry, best first."""
        return self.top(limit=-1)

    def rank_of(self, entry_id: int) -> Optional[int]:
        """1-based standing. Ties break by who got there first."""
        row = self._db.execute("SELECT score, achieved_at FROM leaderboard WHERE id = ?",
                               (entry_id,)).fetchone()
        if row is None:
            return None
        better = self._db.execute(
            "SELECT COUNT(*) FROM leaderboard WHERE score > ? "
            "OR (score = ? AND (achieved_at < ? OR (achieved_at = ? AND id < ?)))",
            (row["score"], row["score"], row["achieved_at"], row["achieved_at"],
             entry_id)).fetchone()[0]
        return int(better) + 1

    def total_players(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM leaderboard").fetchone()[0])

    # ------------------------------------------------------------- settings

    def load_settings(self) -> Settings:
        """Saved settings, clamped; defaults for anything never saved."""
        data = {}
        for row in self._db.execute("SELECT key, value FROM settings"):
            try:
                data[row["key"]] = json.loads(row["value"])
            except (TypeError, ValueError):
                continue
        return Settings.from_dict(data)

    def save_settings(self, settings: Settings) -> None:
        with self._db:
            self._db.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(k, json.dumps(v)) for k, v in settings.to_dict().items()])
            self._bump(_SETTINGS_REV)

    # ----------------------------------------------------------------- live

    def set_live(self, phase: str, player_name: Optional[str] = None, score: int = 0,
                 lives: int = 0, max_lives: int = 0) -> None:
        with self._db:
            self._db.execute(
                "UPDATE live_state SET phase = ?, player_name = ?, score = ?, lives = ?, "
                "max_lives = ?, updated_at = ? WHERE id = 1",
                (phase, player_name, int(score), int(lives), int(max_lives), time.time()))

    def announce(self, entry_id: int) -> None:
        """Ask the leaderboard to flash an entry - call after `submit_score`."""
        with self._db:
            self._db.execute(
                "UPDATE live_state SET highlight_id = ?, highlight_seq = highlight_seq + 1, "
                "updated_at = ? WHERE id = 1", (entry_id, time.time()))

    def get_live(self) -> Live:
        row = self._db.execute("SELECT * FROM live_state WHERE id = 1").fetchone()
        if row is None:
            return Live()
        return Live(row["phase"], row["player_name"], row["score"], row["lives"],
                    row["max_lives"], row["highlight_id"], row["highlight_seq"],
                    row["updated_at"])
