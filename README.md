# Searchlight

A two-screen arcade game where players keep their head inside a wandering square on a camera feed. Leave the square longer than the grace time and you lose a life. The square shrinks and speeds up as you survive, and your score is the time you lasted.

```
  ┌─────────────────────┐        ┌─────────────────────┐
  │  display 0          │        │  display 1          │
  │  game + camera      │◀──DB──▶│  leaderboard        │
  └─────────────────────┘        └─────────────────────┘
```

## Setup

```bash
git clone https://github.com/jonathango98/searchlight.git
cd searchlight
./install.sh
```

`install.sh` does everything once: finds a supported Python, creates `.venv`, installs the packages, downloads the MediaPipe models (~9 MB) and checks the result. It is safe to re-run.

**Python 3.10–3.12 is required (3.11 recommended).** mediapipe 0.10 needs numpy 1.x, which has no builds for Python 3.13 or newer. The installer skips a newer default `python3` and looks for `python3.11`, `python3.12` or `python3.10`. On a Mac with Homebrew and none of those, it runs `brew install python@3.11` for you. Otherwise it says what to install. To choose the interpreter yourself: `PYTHON=/path/to/python3.11 ./install.sh`.

On macOS the terminal needs camera access: **System Settings → Privacy & Security → Camera**. Without it the capture silently yields black frames rather than raising an error, so check there first if the preview stays dark.

## Running

```bash
.venv/bin/python run_searchlight.py              # both windows, cascaded (default)
.venv/bin/python run_searchlight.py --dev        # mouse stands in for your head (testing)
.venv/bin/python run_searchlight.py --fullscreen # kiosk mode: game on display 0, board on display 1
.venv/bin/python run_searchlight.py --fullscreen --game-display 1 --board-display 0  # swap displays
```

Window positioning in windowed mode (the default):
- `--game-pos X,Y` and `--board-pos X,Y`: position each window on your laptop.
- `--game-size WxH` and `--board-size WxH`: resize each window.

`--recordings DIR` sets where run videos are saved (default `recordings/`). See [Recording runs](#recording-runs).

Both windows can be resized by dragging, and **F11** or **Cmd/Ctrl+F** toggles fullscreen at any time.

Quit with **Ctrl/Cmd+Q**, or **Esc** on the attract screen. Closing either window shuts the other down.

## Controls

**Attract screen (waiting for a player):** Press **Tab** or click the gear icon to open the settings menu.

**Name entry:** Type to enter a name. **Enter** to confirm, **Backspace** to delete characters.

**Find player (waiting for head detection):** Stay in the start square until it locks on. You'll see the countdown start when you're ready.

**Play:** Keep your head in the square. Leave it and the grace timer starts—get back in before it expires or you lose a life. Losing a life pauses the run (the square and the clock stop) until you're back in the square, then a short countdown resumes play; leaving during that countdown restarts it. After three lives (adjustable), the game ends.

**Settings:** Arrow keys to navigate, left/right or click to change values. **Enter** to confirm, **Esc** or click out to cancel. The menu shows whether each change takes effect live or at the next run.

## Settings

All settings are stored in the database and persist across runs. A settings menu in the attract screen (Tab or gear icon) lets you change them during play.

| Group | Setting | Range | Default | Notes |
|-------|---------|-------|---------|-------|
| **Gameplay** | Lives | 1–9 | 3 | How many times you can leave the square before game over |
| | Grace time | 0.1–2.0 s | 0.35 s | How long you can be outside before losing a life |
| | Respawn countdown | 1–5 s | 2 s | After losing a life the run pauses until you're back in the square, then counts down this long |
| | Lock-on time | 0.3–3.0 s | 1.0 s | How long your head must stay in the start square before countdown |
| | Countdown | 1–5 s | 3 s | Length of the 3-2-1 before the square starts moving |
| **Square** | Starting size | 20–70% | 42% | Square side as a share of frame height |
| | Smallest size | 8–50% | 16% | The square never shrinks below this |
| | Shrink rate | 0–2% /s | 0.4% /s | How fast the square shrinks |
| | Starting speed | 0–80% /s | 10% /s | Initial movement speed (frame heights per second) |
| | Top speed | 5–120% /s | 45% /s | Maximum movement speed |
| | Speed-up rate | 0–3% /s² | 0.6% /s² | Acceleration |
| | Wobble | 0–4.0 | 1.6 | How sharply the square changes direction; 0 = straight lines |
| **Detection** | Detect mode | Face / Upper body | Face | Face is precise up close; Upper body works from further away |
| | Long-range assist | ON / OFF | ON | Zoom around the player to find small, distant faces |
| | Camera zoom | 1.0–3.0 x | 1.0 x | Crop into the centre of the camera for distant players |
| | Detection confidence | 0.2–0.9 | 0.5 | Lower finds more faces but may mistake other things for one |
| | Smoothing | 0–0.9 | 0.4 | Steadies a jittery head marker; higher adds a little lag |
| | Camera resolution | 960×540 / 1280×720 / 1920×1080 | 1280×720 | Higher helps distant players but costs frame rate |
| | Camera | 0–5 | 0 | Which camera to use if more than one is connected |
| | Mirror | ON / OFF | ON | Flip the picture so moving left moves you left on screen |
| | Show tracking box | ON / OFF | ON | Outline the player's head so they see where the camera is looking |
| **Recording** | Record runs | ON / OFF | OFF | Save the raw camera video of each run (see below) |
| **Leaderboard** | Rows shown | 5–15 | 10 | How many players the leaderboard displays |

**Presets:** The menu includes **Easy**, **Normal**, and **Hard** difficulty buttons that apply sensible defaults for gameplay and square settings, without changing detection or display options.

## Recording runs

Turn on **Settings → Recording → Record runs** to save the raw camera feed of every run, one file per run:

```
recordings/2026-09-28_14-32-05_ALEX.mp4
```

- **Raw** means straight off the camera: full capture resolution, not mirrored, not zoomed, with no square, HUD, or tracking box drawn on it.
- Recording starts at the first countdown and stops on the result screen. If the player drops out during the countdown and locks on again, recording continues in the same file. A player who leaves before the run starts leaves no file behind. Quitting mid-run keeps what was recorded.
- The video is H.264 at a constant 30 fps, timed by capture time, so it plays back at real speed even when tracking runs slower than the camera.
- Encoding runs on its own thread and never slows tracking. If the disk can't keep up, frames are dropped rather than buffered.

Files are never deleted automatically. At 1280×720 expect roughly 5-15 MB per minute.

## Leaderboard editing

Press **E** on the leaderboard screen to enter edit mode. You can:
- Add new player entries
- Rename a player (renaming onto an existing name merges them and keeps the higher score)
- Change a score
- Delete an entry
- Clear all entries

Press **E** again or click to exit edit mode.

## Playing from further away

The game supports three ways to detect heads at distance:

1. **Face mode + long-range assist** (default): Uses MediaPipe BlazeFace with ROI tracking and a tiled zoomed search. Good up to ~10 feet.
2. **Camera zoom**: Crop into the centre of the camera with `--zoom 1.5` or the settings menu. Works well for fixed-position kiosks.
3. **Upper body mode**: Switch to `Detection → Detect → Upper body` in the settings menu. Uses MediaPipe PoseLandmarker to track the shoulders and head together. Works up to ~15 feet but needs enough of the player in frame.

Verify detection at distance before an event using the diagnostic:

```bash
.venv/bin/python scripts/check_camera.py --mode face|upper_body [--no-long-range] [--zoom 1.5] --seconds 20
```

Wave at the camera or have a friend stand far back. You should see live metrics move and detections fire. If detection is weak, adjust the settings.

## Layout

```
run_searchlight.py        starts both windows, ties lifetimes together
searchlight/
  settings.py             every tunable parameter, defined once (pure)
  store.py                SQLite: leaderboard (one row per name), settings, live state
  square.py               wandering square (pure)
  rules.py                lives, grace, respawn, survival clock (pure)
  camera.py               capture thread: mirror, zoom, tracking, smoothing
  recording.py            writes the raw camera feed of a run to video
  tracking/               face detector (+ long-range assist) and pose tracker; pure geometry
  ui/                     theme, window, widgets
  settings_menu.py        in-game settings overlay
  fx.py                   particles, tweens, transitions
  game.py                 game window
  leaderboard.py          leaderboard window + edit mode
scripts/
  fetch_models.sh         downloads MediaPipe models
  fetch_fonts.sh          downloads OFL fonts from google/fonts
  check_camera.py         terminal diagnostic for detection at distance
  render_screens.py       renders every game screen to PNG (development)
assets/fonts/             bundled OFL fonts (system fallback if missing)
tests/                    pytest suite
```

The pure game logic (`settings.py`, `square.py`, `rules.py`, and `tracking/`) imports no pygame, OpenCV, or MediaPipe, so it can be tested against synthetic signals with no camera.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

## Two processes, one database

The windows are separate processes so each can claim a display with the plain `set_mode(display=N)` call, instead of pygame's awkward multi-window extension. They share only `searchlight.db` (SQLite in WAL mode): the game writes live state ten times a second, the leaderboard polls it. Either can be restarted without disturbing the other.

## Environment notes

Two version constraints are load-bearing, both pinned in `requirements.txt`:

- **mediapipe must be < 1.0.** The 1.x face detector crashes on macOS arm64 (`DrishtiMetalHelper ... Service is unavailable`) as the graph opens.
- **opencv must be the headless build.** The regular wheels bundle their own SDL2, whose Objective-C classes collide with pygame's; the runtime binds duplicate class names to whichever registers first. Both entry points import pygame before anything that pulls in cv2, so pygame's SDL wins. mediapipe depends on the regular build, so `install.sh` removes it after installing. From 4.13 the headless macOS wheels bundle SDL2 as well, so opencv is pinned below 4.12.
