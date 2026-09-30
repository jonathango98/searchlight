"""Tests for searchlight.settings.Settings and the settings schema."""

import pytest

from searchlight.settings import (
    BOOL,
    CHOICE,
    FLOAT,
    GROUPS,
    INT,
    LIVE,
    NEXT_RUN,
    PERCENT,
    PLAIN,
    SECONDS,
    Setting,
    SETTINGS,
    SETTINGS_BY_KEY,
    Settings,
    apply_preset,
    matching_preset,
)
from searchlight.tracking.base import MODE_FACE, MODE_UPPER_BODY


class TestSettingsDefaults:
    """Test Settings default values."""

    def test_settings_default_instance(self):
        s = Settings()
        assert s.lives == 3
        assert s.grace_seconds == 0.35
        assert s.start_side == 0.42
        assert s.mode == MODE_FACE

    def test_settings_from_empty_dict(self):
        s = Settings.from_dict({})
        assert s == Settings()

    def test_settings_spec_default_matches_field_default(self):
        """Every Setting's default should match the Settings field default."""
        for spec in SETTINGS:
            settings_value = getattr(Settings(), spec.key)
            assert settings_value == spec.default, f"{spec.key}: Settings field default doesn't match spec"


class TestSettingsSchema:
    """Test that Settings and SETTINGS schema are consistent."""

    def test_every_settings_field_has_spec(self):
        """Every Settings field should have a matching Setting in SETTINGS."""
        settings_fields = {f.name for f in Settings.__dataclass_fields__.values()}
        spec_keys = {s.key for s in SETTINGS}

        # All fields except dataclass metadata should be in SETTINGS
        missing = settings_fields - spec_keys
        assert not missing, f"Settings fields without spec: {missing}"

    def test_every_spec_has_settings_field(self):
        """Every Setting should have a matching field in Settings."""
        settings_fields = {f.name for f in Settings.__dataclass_fields__.values()}
        spec_keys = {s.key for s in SETTINGS}

        extra = spec_keys - settings_fields
        assert not extra, f"Settings specs without field: {extra}"

    def test_all_groups_in_groups_tuple(self):
        """All groups in SETTINGS should be in GROUPS."""
        spec_groups = {s.group for s in SETTINGS}
        for group in spec_groups:
            assert group in GROUPS, f"Group '{group}' not in GROUPS"

    def test_settings_by_key_complete(self):
        """SETTINGS_BY_KEY should have every Setting."""
        for spec in SETTINGS:
            assert spec.key in SETTINGS_BY_KEY
            assert SETTINGS_BY_KEY[spec.key] == spec


class TestClamping:
    """Test value clamping and parsing."""

    def test_clamp_int_in_range(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.clamp(3) == 3
        assert spec.clamp(1) == 1
        assert spec.clamp(9) == 9

    def test_clamp_int_below_min(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.clamp(0) == 1  # min=1

    def test_clamp_int_above_max(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.clamp(10) == 9  # max=9

    def test_clamp_int_rounds_float(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.clamp(3.6) == 4

    def test_clamp_float_in_range(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        assert spec.clamp(0.35) == 0.35

    def test_clamp_float_below_min(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        assert spec.clamp(0.05) == 0.10  # min=0.10

    def test_clamp_float_above_max(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        assert spec.clamp(3.0) == 2.0  # max=2.0

    def test_clamp_string_to_float(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        assert spec.clamp("0.5") == 0.5

    def test_clamp_invalid_string_returns_default(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        assert spec.clamp("abc") == spec.default

    def test_clamp_nan_returns_default(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        assert spec.clamp(float('nan')) == spec.default

    def test_clamp_bool_true(self):
        spec = SETTINGS_BY_KEY["long_range"]
        assert spec.clamp(True) is True
        assert spec.clamp(1) is True

    def test_clamp_bool_false(self):
        spec = SETTINGS_BY_KEY["long_range"]
        assert spec.clamp(False) is False
        assert spec.clamp(0) is False

    def test_clamp_bool_from_string(self):
        spec = SETTINGS_BY_KEY["long_range"]
        assert spec.clamp("true") is True
        assert spec.clamp("TRUE") is True
        assert spec.clamp("yes") is True
        assert spec.clamp("on") is True
        assert spec.clamp("1") is True
        assert spec.clamp("false") is False
        assert spec.clamp("no") is False
        assert spec.clamp("off") is False
        assert spec.clamp("0") is False

    def test_clamp_choice_valid(self):
        spec = SETTINGS_BY_KEY["mode"]
        assert spec.clamp("face") == "face"
        assert spec.clamp("upper_body") == "upper_body"

    def test_clamp_choice_invalid_returns_default(self):
        spec = SETTINGS_BY_KEY["mode"]
        assert spec.clamp("invalid") == spec.default

    def test_clamp_float_snaps_to_step_grid(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        # step=0.05
        clamped = spec.clamp(0.57)  # Not on grid
        # Should snap to nearest grid point
        assert abs(clamped - 0.55) < 0.01 or abs(clamped - 0.60) < 0.01


class TestFromDict:
    """Test Settings.from_dict behavior."""

    def test_from_dict_unknown_keys_ignored(self):
        s = Settings.from_dict({"unknown_key": 999, "lives": 5})
        assert s.lives == 5
        assert s.grace_seconds == Settings().grace_seconds

    def test_from_dict_with_changes_unknown_key_raises(self):
        s = Settings()
        with pytest.raises(KeyError):
            s.with_changes(unknown_key=999)

    def test_from_dict_partial_values(self):
        s = Settings.from_dict({"lives": 5, "grace_seconds": 0.5})
        assert s.lives == 5
        assert s.grace_seconds == 0.5
        assert s.start_side == Settings().start_side


class TestConsistency:
    """Test consistency rules that fix contradictory values."""

    def test_consistency_min_side_clamped_to_start_side(self):
        # Use with_changes to trigger consistency checks
        s = Settings().with_changes(start_side=0.30, min_side=0.40)
        # min_side should be pulled down to start_side
        assert s.min_side == 0.30

    def test_consistency_max_speed_raised_to_start_speed(self):
        # Use with_changes to trigger consistency checks
        s = Settings().with_changes(start_speed=0.20, max_speed=0.10)
        # max_speed should be raised to start_speed
        assert s.max_speed == 0.20

    def test_consistency_not_applied_for_valid_pairs(self):
        s = Settings(start_side=0.42, min_side=0.16, start_speed=0.10, max_speed=0.45)
        assert s.min_side == 0.16
        assert s.max_speed == 0.45


class TestNudge:
    """Test nudge method for arrow key / slider click adjustments."""

    def test_nudge_int_positive(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.nudge(3, 1) == 4

    def test_nudge_int_negative(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.nudge(3, -1) == 2

    def test_nudge_int_stops_at_bounds(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.nudge(9, 1) == 9  # max
        assert spec.nudge(1, -1) == 1  # min

    def test_nudge_float_uses_step(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        # step=0.05
        result = spec.nudge(0.50, 1)
        assert result == 0.55

    def test_nudge_float_negative_steps(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        result = spec.nudge(0.50, -1)
        assert result == 0.45

    def test_nudge_bool_flips_on_odd_steps(self):
        spec = SETTINGS_BY_KEY["long_range"]
        assert spec.nudge(True, 1) is False
        assert spec.nudge(True, 3) is False
        assert spec.nudge(True, 2) is True

    def test_nudge_choice_wraps_forward(self):
        spec = SETTINGS_BY_KEY["mode"]
        result = spec.nudge("face", 1)
        # Should cycle to upper_body
        assert result == "upper_body"

    def test_nudge_choice_wraps_backward(self):
        spec = SETTINGS_BY_KEY["mode"]
        result = spec.nudge("upper_body", -1)
        # Should wrap to face
        assert result == "face"

    def test_nudge_choice_wraps_around(self):
        spec = SETTINGS_BY_KEY["mode"]
        # Step twice forward from upper_body should wrap to face
        result = spec.nudge("upper_body", 1)
        assert result == "face"


class TestFormat:
    """Test format method for menu display."""

    def test_format_seconds(self):
        spec = SETTINGS_BY_KEY["grace_seconds"]
        # display=SECONDS
        assert spec.format(0.35) == "0.35s"
        assert spec.format(0.5) == "0.5s"
        assert spec.format(1.0) == "1s"

    def test_format_percent(self):
        spec = SETTINGS_BY_KEY["start_side"]
        # display=PERCENT
        assert spec.format(0.42) == "42%"
        assert spec.format(0.5) == "50%"

    def test_format_bool_on(self):
        spec = SETTINGS_BY_KEY["long_range"]
        assert spec.format(True) == "ON"

    def test_format_bool_off(self):
        spec = SETTINGS_BY_KEY["long_range"]
        assert spec.format(False) == "OFF"

    def test_format_choice(self):
        spec = SETTINGS_BY_KEY["mode"]
        assert spec.format("face") == "Face"
        assert spec.format("upper_body") == "Upper body"

    def test_format_int_with_unit(self):
        spec = SETTINGS_BY_KEY["countdown_seconds"]
        assert spec.format(3) == "3s"

    def test_format_int_no_unit(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.format(3) == "3"


class TestFraction:
    """Test fraction and from_fraction for slider positioning."""

    def test_fraction_at_min(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.fraction(1) == 0.0

    def test_fraction_at_max(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.fraction(9) == 1.0

    def test_fraction_midpoint(self):
        spec = SETTINGS_BY_KEY["lives"]
        assert spec.fraction(5) == 0.5

    def test_from_fraction_at_zero(self):
        spec = SETTINGS_BY_KEY["lives"]
        result = spec.from_fraction(0.0)
        assert result == 1  # min

    def test_from_fraction_at_one(self):
        spec = SETTINGS_BY_KEY["lives"]
        result = spec.from_fraction(1.0)
        assert result == 9  # max

    def test_from_fraction_midpoint(self):
        spec = SETTINGS_BY_KEY["lives"]
        result = spec.from_fraction(0.5)
        assert result == 5

    def test_fraction_from_fraction_roundtrip(self):
        spec = SETTINGS_BY_KEY["lives"]
        for value in [1, 3, 5, 9]:
            frac = spec.fraction(value)
            restored = spec.from_fraction(frac)
            assert abs(restored - value) < 1


class TestPresets:
    """Test preset application and matching."""

    def test_apply_preset_easy(self):
        s = Settings()
        easy = apply_preset(s, "Easy")

        assert easy.lives == 5
        assert easy.grace_seconds == 0.6
        assert easy.start_side == 0.50

    def test_apply_preset_hard(self):
        s = Settings()
        hard = apply_preset(s, "Hard")

        assert hard.lives == 2
        assert hard.grace_seconds == 0.25
        assert hard.turn_rate == 2.0

    def test_apply_preset_normal(self):
        s = Settings()
        normal = apply_preset(s, "Normal")

        # Normal should equal defaults
        assert normal == Settings()

    def test_matching_preset_normal(self):
        s = Settings()
        assert matching_preset(s) == "Normal"

    def test_matching_preset_easy(self):
        s = Settings()
        easy = apply_preset(s, "Easy")
        assert matching_preset(easy) == "Easy"

    def test_matching_preset_hard(self):
        s = Settings()
        hard = apply_preset(s, "Hard")
        assert matching_preset(hard) == "Hard"

    def test_matching_preset_custom_returns_none(self):
        # A custom value that doesn't match any preset
        s = Settings().with_changes(lives=4)
        assert matching_preset(s) is None

    def test_preset_never_changes_detection_keys(self):
        s = Settings().with_changes(mode="upper_body", min_confidence=0.7)
        easy = apply_preset(s, "Easy")

        # Detection settings should not change
        assert easy.mode == "upper_body"
        assert easy.min_confidence == 0.7


class TestSquareConfig:
    """Test square_config method."""

    def test_square_config_default_aspect(self):
        s = Settings()
        cfg = s.square_config()

        assert cfg.aspect == 16 / 9
        assert cfg.start_side == s.start_side
        assert cfg.min_side == s.min_side

    def test_square_config_custom_aspect(self):
        s = Settings()
        cfg = s.square_config(aspect=1.0)

        assert cfg.aspect == 1.0

    def test_square_config_carries_all_values(self):
        s = Settings().with_changes(
            start_side=0.5,
            min_side=0.2,
            shrink_per_second=0.01,
            start_speed=0.14,  # Snaps to step grid (0.02)
            max_speed=0.5,
            speed_ramp=0.01,
            turn_rate=2.0
        )
        cfg = s.square_config()

        assert cfg.start_side == 0.5
        assert cfg.min_side == 0.2
        assert cfg.shrink_per_second == 0.01
        assert cfg.start_speed == 0.14
        assert cfg.max_speed == 0.5
        assert cfg.speed_ramp == 0.01
        assert cfg.turn_rate == 2.0


class TestRunConfig:
    """Test run_config method."""

    def test_run_config_default(self):
        s = Settings()
        cfg = s.run_config()

        assert cfg.lives == s.lives
        assert cfg.grace_seconds == s.grace_seconds
        assert cfg.respawn_seconds == s.respawn_seconds

    def test_run_config_custom_values(self):
        s = Settings().with_changes(
            lives=5,
            grace_seconds=0.6,
            respawn_seconds=2.0
        )
        cfg = s.run_config()

        assert cfg.lives == 5
        assert cfg.grace_seconds == 0.6
        assert cfg.respawn_seconds == 2.0


class TestCameraOptions:
    """Test camera_options method."""

    def test_camera_options_default(self):
        s = Settings()
        opts = s.camera_options()

        assert opts.camera_index == 0
        assert opts.width == 1280
        assert opts.height == 720
        assert opts.mode == "face"
        assert opts.long_range is True
        assert opts.zoom == 1.0

    def test_camera_options_resolution_parsing(self):
        s = Settings().with_changes(resolution="1920x1080")
        opts = s.camera_options()

        assert opts.width == 1920
        assert opts.height == 1080

    def test_camera_options_resolution_960x540(self):
        s = Settings().with_changes(resolution="960x540")
        opts = s.camera_options()

        assert opts.width == 960
        assert opts.height == 540

    def test_camera_options_carries_all_values(self):
        s = Settings().with_changes(
            camera_index=2,
            zoom=2.0,
            min_confidence=0.7,
            smoothing=0.6,
            mode="upper_body",
            mirror=False
        )
        opts = s.camera_options()

        assert opts.camera_index == 2
        assert opts.zoom == 2.0
        assert opts.min_confidence == 0.7
        assert opts.smoothing == 0.6
        assert opts.mode == "upper_body"
        assert opts.mirror is False


class TestWithChanges:
    """Test with_changes method."""

    def test_with_changes_single_value(self):
        s = Settings()
        modified = s.with_changes(lives=5)

        assert modified.lives == 5
        assert modified.grace_seconds == s.grace_seconds

    def test_with_changes_multiple_values(self):
        s = Settings()
        modified = s.with_changes(lives=5, grace_seconds=0.5, start_side=0.5)

        assert modified.lives == 5
        assert modified.grace_seconds == 0.5
        assert modified.start_side == 0.5

    def test_with_changes_returns_new_instance(self):
        s = Settings()
        modified = s.with_changes(lives=5)

        assert s is not modified
        assert s.lives == 3
        assert modified.lives == 5

    def test_with_changes_clamping_applied(self):
        s = Settings()
        modified = s.with_changes(lives=999)

        # Should be clamped to max=9
        assert modified.lives == 9

    def test_with_changes_unknown_key_raises_keyerror(self):
        s = Settings()
        with pytest.raises(KeyError):
            s.with_changes(unknown_key=123)
