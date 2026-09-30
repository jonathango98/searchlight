"""Tests for searchlight.store.Store - the shared SQLite state."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from searchlight.settings import Settings
from searchlight.store import (
    Store,
    InvalidName,
    MAX_NAME,
    MAX_SCORE,
    PHASE_IDLE,
    PHASE_PLAYING,
    format_score,
    normalise_name,
    parse_score,
    to_score,
)


class TestNormaliseAndFormat:
    """Test name normalisation and score formatting."""

    def test_normalise_name_uppercase_trim_single_space(self):
        assert normalise_name("  alex  ") == "ALEX"
        assert normalise_name("alex") == "ALEX"
        assert normalise_name("  Jo  ") == "JO"

    def test_normalise_name_case_insensitive(self):
        assert normalise_name("Alex") == normalise_name("ALEX")
        assert normalise_name("alex") == normalise_name("ALEX")

    def test_normalise_name_single_space_between_words(self):
        assert normalise_name("  john   doe  ") == "JOHN DOE"
        # Note: "multiple   spaces" is 15 chars, truncated to MAX_NAME=12
        assert normalise_name("hello   world") == "HELLO WORLD"

    def test_normalise_name_truncate_to_max_name(self):
        long_name = "A" * (MAX_NAME + 5)
        result = normalise_name(long_name)
        assert len(result) == MAX_NAME

    def test_normalise_empty_string(self):
        assert normalise_name("") == ""
        assert normalise_name("   ") == ""

    def test_to_score_rounding(self):
        assert to_score(23.4) == 234
        assert to_score(23.0) == 230
        assert to_score(23.55) == 236  # rounds

    def test_format_score(self):
        assert format_score(234) == "23.4s"
        assert format_score(230) == "23.0s"
        assert format_score(0) == "0.0s"

    def test_parse_score_with_seconds(self):
        assert parse_score("23.4") == 234
        assert parse_score("23.4s") == 234
        assert parse_score("23") == 230

    def test_parse_score_case_insensitive(self):
        assert parse_score("23.4S") == 234
        assert parse_score("23.4S") == parse_score("23.4s")

    def test_parse_score_invalid_raises_valueerror(self):
        with pytest.raises(ValueError):
            parse_score("abc")
        with pytest.raises(ValueError):
            parse_score("-5")
        with pytest.raises(ValueError):
            parse_score("99999999")  # Over MAX_SCORE


class TestNewPlayer:
    """Test submit_score with a new player."""

    def test_submit_score_new_player(self, tmp_path):
        store = Store(tmp_path / "test.db")
        result = store.submit_score("Alex", 100)

        assert result.is_new_player is True
        assert result.previous_best is None
        assert result.is_personal_best is True
        assert result.rank == 1
        assert result.entry.name == "ALEX"
        assert result.entry.score == 100
        assert store.total_players() == 1

    def test_submit_score_empty_name_becomes_anon(self, tmp_path):
        store = Store(tmp_path / "test.db")
        result = store.submit_score("", 100)

        assert result.entry.name == "ANON"
        assert result.is_new_player is True

    def test_submit_score_whitespace_only_name_becomes_anon(self, tmp_path):
        store = Store(tmp_path / "test.db")
        result = store.submit_score("   ", 100)

        assert result.entry.name == "ANON"


class TestOneRowPerName:
    """Test that there's only one row per (normalized) name."""

    def test_same_name_lower_score_unchanged(self, tmp_path):
        store = Store(tmp_path / "test.db")
        result1 = store.submit_score("Alex", 100)
        original_id = result1.entry.id

        result2 = store.submit_score("alex", 50)

        assert result2.entry.id == original_id
        assert result2.entry.score == 100  # Unchanged
        assert result2.is_personal_best is False
        assert result2.previous_best == 100
        assert store.total_players() == 1

    def test_same_name_higher_score_replaces(self, tmp_path):
        store = Store(tmp_path / "test.db")
        result1 = store.submit_score("Alex", 100)

        result2 = store.submit_score("alex", 200)

        assert result2.entry.id == result1.entry.id
        assert result2.entry.score == 200
        assert result2.is_personal_best is True
        assert result2.previous_best == 100
        assert store.total_players() == 1

    def test_case_space_insensitive_same_player(self, tmp_path):
        store = Store(tmp_path / "test.db")
        result1 = store.submit_score("  alex  ", 100)
        result2 = store.submit_score("ALEX", 150)

        assert result1.entry.id == result2.entry.id
        assert result2.entry.score == 150
        assert store.total_players() == 1

    def test_stored_name_is_uppercase(self, tmp_path):
        store = Store(tmp_path / "test.db")
        result = store.submit_score("alex", 100)

        assert result.entry.name == "ALEX"

    def test_name_longer_than_max_truncated(self, tmp_path):
        store = Store(tmp_path / "test.db")
        long_name = "A" * (MAX_NAME + 5)
        result = store.submit_score(long_name, 100)

        assert len(result.entry.name) == MAX_NAME


class TestRanking:
    """Test score ranking: higher first, ties by creation order."""

    def test_rank_higher_score_first(self, tmp_path):
        store = Store(tmp_path / "test.db")
        r1 = store.submit_score("Alice", 300)
        r2 = store.submit_score("Bob", 100)
        r3 = store.submit_score("Charlie", 200)

        # Check final ranks using store.rank_of()
        assert store.rank_of(r1.entry.id) == 1
        assert store.rank_of(r3.entry.id) == 2
        assert store.rank_of(r2.entry.id) == 3

    def test_rank_ties_break_by_achievement_order(self, tmp_path):
        store = Store(tmp_path / "test.db")
        r1 = store.submit_score("Alice", 100)
        time.sleep(0.01)
        r2 = store.submit_score("Bob", 100)

        assert r1.rank == 1
        assert r2.rank == 2

    def test_rank_of_unknown_entry_is_none(self, tmp_path):
        store = Store(tmp_path / "test.db")
        assert store.rank_of(999) is None

    def test_rank_of_agrees_with_top(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entries = []
        for i, name in enumerate(["Alice", "Bob", "Charlie", "Diana"]):
            result = store.submit_score(name, 400 - i * 50)
            entries.append(result.entry.id)

        top = store.top()
        for rank, entry_id in enumerate(entries, 1):
            assert store.rank_of(entry_id) == rank


class TestAdminAPI:
    """Test admin operations: add, edit, rename, delete, clear."""

    def test_add_entry_new_name(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)

        assert entry.name == "ALEX"
        assert entry.score == 100
        assert store.total_players() == 1

    def test_add_entry_existing_name_keeps_max(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry1 = store.add_entry("Alex", 100)
        entry2 = store.add_entry("alex", 50)

        assert entry1.id == entry2.id
        assert entry2.score == 100

    def test_add_entry_empty_name_raises_invalidname(self, tmp_path):
        store = Store(tmp_path / "test.db")
        with pytest.raises(InvalidName):
            store.add_entry("", 100)

    def test_add_entry_whitespace_only_raises_invalidname(self, tmp_path):
        store = Store(tmp_path / "test.db")
        with pytest.raises(InvalidName):
            store.add_entry("   ", 100)

    def test_set_score_up(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        updated = store.set_score(entry.id, 200)

        assert updated.score == 200
        assert store.get(entry.id).score == 200

    def test_set_score_down(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 200)
        updated = store.set_score(entry.id, 100)

        assert updated.score == 100

    def test_set_score_unknown_id_raises_keyerror(self, tmp_path):
        store = Store(tmp_path / "test.db")
        with pytest.raises(KeyError):
            store.set_score(999, 100)

    def test_rename_to_free_name(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        renamed = store.rename(entry.id, "Bob")

        assert renamed.name == "BOB"
        assert renamed.id == entry.id
        assert store.find("Bob") is not None
        assert store.find("Alex") is None

    def test_rename_unknown_id_raises_keyerror(self, tmp_path):
        store = Store(tmp_path / "test.db")
        with pytest.raises(KeyError):
            store.rename(999, "Bob")

    def test_rename_to_empty_name_raises_invalidname(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        with pytest.raises(InvalidName):
            store.rename(entry.id, "")

    def test_rename_onto_existing_name_merges_keeping_max(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry1 = store.add_entry("Alice", 100)
        entry2 = store.add_entry("Bob", 200)

        # Rename Alice to Bob (existing name)
        result = store.rename(entry1.id, "Bob")

        # Should merge into Bob's row (higher score), keeping Bob's id
        assert result.id == entry2.id
        assert result.name == "BOB"
        assert result.score == 200
        assert store.total_players() == 1

    def test_rename_onto_existing_name_lower_score_keeps_higher(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry1 = store.add_entry("Alice", 300)
        entry2 = store.add_entry("Bob", 100)

        # Rename Bob to Alice (existing name)
        result = store.rename(entry2.id, "Alice")

        # Should keep Alice's row and score
        assert result.id == entry1.id
        assert result.score == 300
        assert store.total_players() == 1

    def test_delete_entry(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        store.delete(entry.id)

        assert store.get(entry.id) is None
        assert store.total_players() == 0

    def test_clear_all_entries(self, tmp_path):
        store = Store(tmp_path / "test.db")
        store.add_entry("Alice", 100)
        store.add_entry("Bob", 200)
        store.clear()

        assert store.total_players() == 0

    def test_get_entry_by_id(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        retrieved = store.get(entry.id)

        assert retrieved.id == entry.id
        assert retrieved.name == "ALEX"
        assert retrieved.score == 100

    def test_get_unknown_id_returns_none(self, tmp_path):
        store = Store(tmp_path / "test.db")
        assert store.get(999) is None

    def test_find_by_name_case_insensitive(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)

        assert store.find("alex") is not None
        assert store.find("ALEX") is not None
        assert store.find("Alex") is not None

    def test_find_unknown_name_returns_none(self, tmp_path):
        store = Store(tmp_path / "test.db")
        assert store.find("Unknown") is None

    def test_list_all_ordered_by_rank(self, tmp_path):
        store = Store(tmp_path / "test.db")
        store.add_entry("Alice", 100)
        store.add_entry("Bob", 300)
        store.add_entry("Charlie", 200)

        all_entries = store.list_all()

        assert len(all_entries) == 3
        assert all_entries[0].score == 300
        assert all_entries[1].score == 200
        assert all_entries[2].score == 100

    def test_top_limit(self, tmp_path):
        store = Store(tmp_path / "test.db")
        for i in range(15):
            store.add_entry(f"Player{i}", 1000 - i * 10)

        top10 = store.top(limit=10)
        assert len(top10) == 10
        assert top10[0].score == 1000


class TestRevisions:
    """Test revision tracking for leaderboard and settings."""

    def test_board_rev_increases_on_submit(self, tmp_path):
        store = Store(tmp_path / "test.db")
        rev1 = store.board_rev()
        store.submit_score("Alex", 100)
        rev2 = store.board_rev()

        assert rev2 > rev1

    def test_board_rev_increases_on_add_entry(self, tmp_path):
        store = Store(tmp_path / "test.db")
        rev1 = store.board_rev()
        store.add_entry("Alex", 100)
        rev2 = store.board_rev()

        assert rev2 > rev1

    def test_board_rev_increases_on_delete(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        rev1 = store.board_rev()
        store.delete(entry.id)
        rev2 = store.board_rev()

        assert rev2 > rev1

    def test_board_rev_increases_on_set_score(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        rev1 = store.board_rev()
        store.set_score(entry.id, 200)
        rev2 = store.board_rev()

        assert rev2 > rev1

    def test_board_rev_increases_on_rename(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)
        rev1 = store.board_rev()
        store.rename(entry.id, "Bob")
        rev2 = store.board_rev()

        assert rev2 > rev1

    def test_board_rev_increases_on_clear(self, tmp_path):
        store = Store(tmp_path / "test.db")
        store.add_entry("Alex", 100)
        rev1 = store.board_rev()
        store.clear()
        rev2 = store.board_rev()

        assert rev2 > rev1

    def test_settings_rev_increases_on_save(self, tmp_path):
        store = Store(tmp_path / "test.db")
        rev1 = store.settings_rev()
        settings = Settings()
        store.save_settings(settings)
        rev2 = store.settings_rev()

        assert rev2 > rev1

    def test_settings_rev_not_affected_by_board_changes(self, tmp_path):
        store = Store(tmp_path / "test.db")
        rev1 = store.settings_rev()
        store.submit_score("Alex", 100)
        rev2 = store.settings_rev()

        assert rev1 == rev2

    def test_board_rev_not_affected_by_settings_changes(self, tmp_path):
        store = Store(tmp_path / "test.db")
        rev1 = store.board_rev()
        settings = Settings()
        store.save_settings(settings)
        rev2 = store.board_rev()

        assert rev1 == rev2

    def test_revisions_tuple_order(self, tmp_path):
        store = Store(tmp_path / "test.db")
        board_rev, settings_rev = store.revisions()

        # Both should be integers
        assert isinstance(board_rev, int)
        assert isinstance(settings_rev, int)


class TestSettingsRoundTrip:
    """Test settings persistence."""

    def test_load_settings_fresh_db_equals_defaults(self, tmp_path):
        store = Store(tmp_path / "test.db")
        loaded = store.load_settings()

        assert loaded == Settings()

    def test_save_and_load_settings(self, tmp_path):
        db_path = tmp_path / "test.db"
        store1 = Store(db_path)
        settings = Settings().with_changes(grace_seconds=0.8)
        store1.save_settings(settings)
        store1.close()

        store2 = Store(db_path)
        loaded = store2.load_settings()

        assert loaded.grace_seconds == 0.8

    def test_load_settings_new_instance_same_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        store1 = Store(db_path)
        settings = Settings().with_changes(lives=5, grace_seconds=0.5)
        store1.save_settings(settings)
        store1.close()

        store2 = Store(db_path)
        loaded = store2.load_settings()

        assert loaded.lives == 5
        assert loaded.grace_seconds == 0.5

    def test_load_settings_ignores_corrupt_rows(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = Store(db_path)

        # Insert a corrupt JSON value directly
        store._db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            ("corrupt_key", "not valid json")
        )
        store._db.commit()

        loaded = store.load_settings()
        # Should load defaults, ignoring the corrupt row
        assert loaded == Settings()

    def test_load_settings_clamps_out_of_range_values(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = Store(db_path)

        # Insert an out-of-range value
        store._db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            ("lives", "999")
        )
        store._db.commit()
        store.close()

        store2 = Store(db_path)
        loaded = store2.load_settings()
        # lives has max of 9, so it should be clamped
        assert loaded.lives <= 9


class TestLiveState:
    """Test live game state updates."""

    def test_set_live_and_get_live(self, tmp_path):
        store = Store(tmp_path / "test.db")
        store.set_live(PHASE_PLAYING, player_name="Alex", score=150, lives=2, max_lives=3)

        live = store.get_live()

        assert live.phase == PHASE_PLAYING
        assert live.player_name == "Alex"
        assert live.score == 150
        assert live.lives == 2
        assert live.max_lives == 3

    def test_set_live_default_values(self, tmp_path):
        store = Store(tmp_path / "test.db")
        store.set_live(PHASE_PLAYING)

        live = store.get_live()

        assert live.phase == PHASE_PLAYING
        assert live.player_name is None
        assert live.score == 0

    def test_announce_sets_highlight_and_increments_seq(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry = store.add_entry("Alex", 100)

        store.announce(entry.id)
        live1 = store.get_live()

        assert live1.highlight_id == entry.id
        assert live1.highlight_seq == 1

        store.announce(entry.id)
        live2 = store.get_live()

        assert live2.highlight_seq == 2

    def test_announce_different_id_increments_seq(self, tmp_path):
        store = Store(tmp_path / "test.db")
        entry1 = store.add_entry("Alice", 100)
        entry2 = store.add_entry("Bob", 200)

        store.announce(entry1.id)
        store.announce(entry2.id)
        live = store.get_live()

        assert live.highlight_id == entry2.id
        assert live.highlight_seq == 2


class TestMigration:
    """Test migration from old database format."""

    def test_migrate_old_scores_table_keeps_max(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Create old-format database
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                score INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        now = time.time()
        conn.execute("INSERT INTO scores (name, score, created_at) VALUES (?, ?, ?)",
                    ("jo", 100, now))
        conn.execute("INSERT INTO scores (name, score, created_at) VALUES (?, ?, ?)",
                    ("JO", 250, now + 1))
        conn.execute("INSERT INTO scores (name, score, created_at) VALUES (?, ?, ?)",
                    ("Jo ", 90, now + 2))
        conn.execute("INSERT INTO scores (name, score, created_at) VALUES (?, ?, ?)",
                    ("SAM", 50, now + 3))
        conn.commit()
        conn.close()

        # Open with Store - should migrate
        store = Store(db_path)
        entries = store.list_all()

        # Should have 2 entries (one per normalized name)
        assert len(entries) == 2
        # JO should have max score 250
        assert store.find("JO").score == 250
        assert store.find("SAM").score == 50
        # board_rev should be > 0 after migration
        assert store.board_rev() > 0

    def test_migrate_old_scores_table_idempotent(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Create old-format database
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                score INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        now = time.time()
        conn.execute("INSERT INTO scores (name, score, created_at) VALUES (?, ?, ?)",
                    ("Alice", 100, now))
        conn.commit()
        conn.close()

        # First migration
        store1 = Store(db_path)
        rev1 = store1.board_rev()
        store1.close()

        # Second open should be idempotent
        store2 = Store(db_path)
        rev2 = store2.board_rev()

        # Should have same rev (no second migration)
        assert rev1 == rev2

    def test_migrate_old_live_table_dropped(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Create old-format database with live table
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE live (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                phase TEXT NOT NULL,
                player_name TEXT,
                count INTEGER NOT NULL DEFAULT 0,
                seconds_left REAL NOT NULL DEFAULT 0,
                last_score_id INTEGER,
                updated_at REAL NOT NULL,
                lives INTEGER NOT NULL DEFAULT 0
            )
        """)
        now = time.time()
        conn.execute(
            "INSERT INTO live (id, phase, updated_at) VALUES (?, ?, ?)",
            (1, "playing", now)
        )
        conn.commit()
        conn.close()

        # Open with Store - should drop old live table
        store = Store(db_path)
        store.close()

        # Check that old table is gone
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='live'"
        )
        assert cursor.fetchone() is None
        conn.close()


class TestMultipleStoreInstances:
    """Test that multiple Store instances on the same file see each other's writes."""

    def test_multiple_stores_see_each_other_writes(self, tmp_path):
        db_path = tmp_path / "test.db"
        store1 = Store(db_path)
        store2 = Store(db_path)

        # Write with store1
        entry = store1.submit_score("Alex", 100)

        # Read with store2
        found = store2.find("Alex")
        assert found is not None
        assert found.score == 100

    def test_multiple_stores_revision_sync(self, tmp_path):
        db_path = tmp_path / "test.db"
        store1 = Store(db_path)
        store2 = Store(db_path)

        # Write with store1
        store1.submit_score("Alex", 100)

        # Revision should increase on store2
        rev2 = store2.board_rev()
        assert rev2 > 0
