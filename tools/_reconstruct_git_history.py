# Reconstruct narrative git history from Cursor chat milestones.
# All commits use the FINAL working tree (true time-travel is impossible).
# Run from repo root: python tools/_reconstruct_git_history.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

GITIGNORE = r"""# Python / Django
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/

# Local env
venv/
.venv/
env/
.env
.env.*

# macOS / junk
.DS_Store
__MACOSX/

# Django local runtime
*.sqlite3
fatigue_detection_system/db.sqlite3
fatigue_detection_system/media/results/*
!fatigue_detection_system/media/results/.gitkeep
fatigue_detection_system/media/uploads/*
!fatigue_detection_system/media/uploads/.gitkeep

# Training / duplicate weights
runs/
fatigue_detection_system/weights/*.bz2
fatigue_detection_system/weights/last.pt

# Bulky installers (keep paths documented in deploy/)
*.exe
!deploy/*.bat

# Agent temp scratch (should not exist)
_patch_*.py
_dump_*.py
**/_patch_*.py
**/_dump_*.py
**/_fix_*.py
**/_test_fatigue_*.py
**/_process_image_snippet.py

# IDE
.idea/
.vscode/
*.swp
"""


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", *args], check=check)


def commit(title: str, body: str) -> None:
    # Ensure something is staged
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not staged:
        print(f"SKIP (nothing staged): {title}")
        return
    # PowerShell-safe: pass message via -m twice
    git("commit", "-m", title, "-m", body)
    print(f"OK: {title}")


def add_paths(paths: list[str]) -> None:
    existing = []
    for p in paths:
        pp = ROOT / p
        if pp.exists():
            existing.append(p)
        else:
            print(f"  missing (skip add): {p}")
    if not existing:
        return
    git("add", "-A", "--", *existing)


def ensure_gitkeep(rel: str) -> None:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")


def main() -> int:
    if not (ROOT / "fatigue_detection_system" / "manage.py").exists():
        print("manage.py not found", file=sys.stderr)
        return 1

    ensure_gitkeep("fatigue_detection_system/media/results/.gitkeep")
    ensure_gitkeep("fatigue_detection_system/media/uploads/.gitkeep")

    (ROOT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    if not (ROOT / ".git").exists():
        git("init")
        git("branch", "-M", "main")

    # Avoid committing if user already has commits? allow rebuild only on empty
    log = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT,
        capture_output=True,
    )
    if log.returncode == 0:
        print("ERROR: repo already has commits. Aborting to avoid rewriting history.")
        print("Delete .git first if you want to re-run reconstruction.")
        return 2

    # ---------- 1 ----------
    add_paths([".gitignore"])
    commit(
        "chore: add gitignore for env, media, installers, and scratch files",
        "Reconstructed history bootstrap. Excludes venv, sqlite, uploads/results, "
        "training runs, bulky .exe installers, and agent temp patch scripts.\n\n"
        "Source: reconstructed from Cursor chats (afcf/1b48/0e81/ad41).",
    )

    # ---------- 2 baseline ----------
    baseline = [
        "README.md",
        "requirements.txt",
        "data.yaml",
        "fatigue_detection_system/manage.py",
        "fatigue_detection_system/generate_sounds.py",
        "fatigue_detection_system/accounts",
        "fatigue_detection_system/fatigue_detection",
        "fatigue_detection_system/static",
        "fatigue_detection_system/templates",
        "fatigue_detection_system/webfonts",
        "fatigue_detection_system/media/results/.gitkeep",
        "fatigue_detection_system/media/uploads/.gitkeep",
        "fatigue_detection_system/detection/__init__.py",
        "fatigue_detection_system/detection/admin.py",
        "fatigue_detection_system/detection/apps.py",
        "fatigue_detection_system/detection/tests.py",
        "fatigue_detection_system/detection/migrations/__init__.py",
        "fatigue_detection_system/detection/migrations/0001_initial.py",
        "fatigue_detection_system/detection/migrations/0002_detectionsession_status.py",
        "fatigue_detection_system/detection/templates/detection/dashboard.html",
        "fatigue_detection_system/detection/templates/detection/history.html",
        "fatigue_detection_system/detection/templates/detection/realtime_detection.html",
        "fatigue_detection_system/detection/templates/detection/statistics.html",
        "fatigue_detection_system/detection/templates/detection/video_upload.html",
    ]
    # Chinese readme if present
    for name in ROOT.iterdir():
        if name.suffix.lower() == ".md" and name.name not in ("README.md",):
            baseline.append(name.name)
    add_paths(baseline)
    commit(
        "chore: import baseline Django fatigue detection app",
        "Initial project tree as it existed before Cursor agent sessions: "
        "accounts, settings, static assets, core detection migrations 0001-0002, "
        "and non-realtime templates.\n\n"
        "NOTE: This is a reconstructed baseline from the final workspace; "
        "pre-chat exact bytes are unavailable.\n\n"
        "Chats: pre-afcf / pre-1b48.",
    )

    # ---------- 3 weights ----------
    add_paths(
        [
            "fatigue_detection_system/weights/best.pt",
            "fatigue_detection_system/weights/shape_predictor_68_face_landmarks.dat",
        ]
    )
    commit(
        "chore: add YOLO and dlib landmark model weights",
        "Ship runtime weights required for face/behavior and EAR/MAR detection "
        "(best.pt + shape_predictor_68_face_landmarks.dat).\n\n"
        "Ignored duplicates: last.pt, .bz2 archives, runs/detect/train.",
    )

    # ---------- 4 detection core models (final models.py without claiming 0003) ----------
    add_paths(["fatigue_detection_system/detection/models.py"])
    commit(
        "feat: add detection domain models (sessions, results, system config)",
        "SystemConfig key-value settings and detection session/result models.\n\n"
        "Later commits add PERCLOS fields via migration 0003.\n\n"
        "Chat touchpoints: 1b48 / ad41 (config keys evolved).",
    )

    # ---------- 5 afcf realtime fix files that are exclusive-ish: stop scripts first ----------
    add_paths(["start.bat", "stop.bat", "stop.ps1"])
    commit(
        "feat: add root start/stop scripts for local Django",
        "One-click start.bat / stop.bat / stop.ps1 to run and kill runserver "
        "processes under this project root.\n\n"
        "Chat: afcf99e5 (Recognition not running issue).",
    )

    # ---------- 6 vendor dlib wheel ----------
    add_paths(["dlib-19.19.0-cp38-cp38-win_amd64.whl"])
    commit(
        "chore: vendor dlib cp38 Windows wheel for offline install",
        "Bundle dlib-19.19.0-cp38-cp38-win_amd64.whl used by portable/one-click installers.\n\n"
        "Chats: 1b48 portable deploy / ad41 install bundle.",
    )

    # ---------- 7 early detector modules (final content; evolved across chats) ----------
    add_paths(
        [
            "fatigue_detection_system/detection/utils/__init__.py",
            "fatigue_detection_system/detection/utils/dlib_detector.py",
            "fatigue_detection_system/detection/utils/yolo_detector.py",
        ]
    )
    commit(
        "feat: add dlib EAR/MAR and YOLO behavior detectors",
        "Core detectors used by webcam and GigE pipelines.\n\n"
        "Content is the FINAL optimized version (spatial filters, per-class conf, "
        "device/FP16 hooks). Intermediate Cursor edits are not separately recoverable.\n\n"
        "Chats: afcf (stabilize) → 1b48 (dlib mouth) → ad41 (rewrite/behavior fixes).",
    )

    # ---------- 8 portable deploy ----------
    add_paths(
        [
            "deploy/prepare_portable.ps1",
            "deploy/start.bat",
            "deploy/PORTABLE_README.txt",
        ]
    )
    commit(
        "feat: add portable package builder for industrial PCs",
        "prepare_portable.ps1 copies Python runtime + venv + app into "
        "SleepyDetect_Portable with start.bat.\n\n"
        "Chat: 1b48f9d0 (Project architecture → portable deploy).",
    )

    # ---------- 9 MVS SDK module ----------
    add_paths(
        [
            "fatigue_detection_system/detection/utils/hik_mvs/__init__.py",
            "fatigue_detection_system/detection/utils/hik_mvs/camera.py",
            "fatigue_detection_system/detection/utils/hik_mvs/grabber.py",
            "fatigue_detection_system/detection/utils/hik_mvs/MvImport",
        ]
    )
    commit(
        "feat: integrate Hikvision MVS GigE backend capture",
        "Add hik_mvs camera/grabber and MvImport bindings for industrial GigE cameras. "
        "Final grabber includes EAR/YOLO worker threads, warmup, and persist policy "
        "from later optimization work.\n\n"
        "Chats: 1b48 (introduce MVS) → ad41 (perf, AE/gain, scheduler, persist).",
    )

    # ---------- 10 MVS deploy docs/helpers ----------
    add_paths(
        [
            "deploy/MVS_SETUP.txt",
            "deploy/copy_mvs_sdk.ps1",
        ]
    )
    commit(
        "docs: add MVS 4.6.3 setup notes and SDK copy helper",
        "Document matching MVS/SDK versions and copy_mvs_sdk.ps1 to refresh MvImport.\n\n"
        "Chat: 1b48f9d0.",
    )

    # ---------- 11 URLs + system config UI (camera source etc.) ----------
    add_paths(
        [
            "fatigue_detection_system/detection/urls.py",
            "fatigue_detection_system/detection/templates/detection/system_config.html",
        ]
    )
    commit(
        "feat: expose MVS/webcam source and detection settings in system config",
        "Wire URLs and system_config UI for camera mode, thresholds, device, "
        "and later scheduler/timing options (final template).\n\n"
        "Chats: 1b48 (source select) → ad41 (timing/frequencies/scheduler).",
    )

    # ---------- 12 mono preprocess ----------
    add_paths(["fatigue_detection_system/detection/utils/mono_preprocess.py"])
    commit(
        "feat: add CLAHE mono-camera preprocess helper",
        "Optional grayscale industrial-camera preprocess before landmarks/YOLO.\n\n"
        "Chat: ad41efed (Optimization suggestions / mono accuracy).",
    )

    # ---------- 13 fatigue tracker + migration ----------
    add_paths(
        [
            "fatigue_detection_system/detection/utils/fatigue_tracker.py",
            "fatigue_detection_system/detection/migrations/0003_detectionresult_perclos_eye_closed_ms.py",
        ]
    )
    commit(
        "feat: add multi-frame fatigue tracker with PERCLOS fields",
        "FatigueTemporalTracker (EAR hysteresis, blink vs microsleep, PERCLOS, yawn "
        "confirm/hold) plus DetectionResult migration for perclos/eye_closed_ms.\n\n"
        "Chat: ad41efed.",
    )

    # ---------- 14 compute scheduler ----------
    add_paths(["fatigue_detection_system/detection/utils/compute_scheduler.py"])
    commit(
        "feat: add CPU/CUDA compute scheduler for EAR and YOLO",
        "Thread quotas, CUDA YOLO + FP16 preference, and EAR yield/stretch when "
        "YOLO is busy on CPU hosts.\n\n"
        "Chat: ad41efed.",
    )

    # ---------- 15 views (final integration hub) ----------
    add_paths(["fatigue_detection_system/detection/views.py"])
    commit(
        "feat: integrate realtime APIs, MVS control, persist, and warmup",
        "Final views.py covering:\n"
        "- session/get_result robustness (afcf)\n"
        "- complete session status + MVS start/status/preview (1b48)\n"
        "- persist_detection_snapshot, reset_fatigue, timing, warmup (ad41)\n\n"
        "Single commit because only the final file exists on disk.",
    )

    # ---------- 16 realtime UI ----------
    add_paths(
        [
            "fatigue_detection_system/detection/templates/detection/realtime.html",
        ]
    )
    commit(
        "feat: upgrade realtime page for detection loop, MVS, and loading UX",
        "Final realtime.html: detection polling fixes (afcf), GigE/webcam source "
        "selection and preview (1b48), timing metrics and 5s loading warmup (ad41).\n\n"
        "Chats: afcf → 1b48 → ad41.",
    )

    # ---------- 17 history/result UI ----------
    add_paths(
        [
            "fatigue_detection_system/detection/templates/detection/results.html",
            "fatigue_detection_system/detection/templates/detection/session_detail.html",
        ]
    )
    commit(
        "fix: show EAR/PERCLOS and behavior evidence in history views",
        "Update results/session_detail templates so history is not stuck as all-normal "
        "and can display tracker metrics.\n\n"
        "Chat: ad41efed (history persistence / landmark miss).",
    )

    # ---------- 18 finetune scaffolding (0e81 intent + ad41 code) ----------
    add_paths(
        [
            "tools/finetune_mono_yolo.py",
            "datasets/mono_behavior",
        ]
    )
    commit(
        "chore: add mono-behavior YOLO finetune scaffolding",
        "datasets/mono_behavior layout + tools/finetune_mono_yolo.py for site-specific "
        "smoke/phone/water fine-tuning on mono industrial frames.\n\n"
        "Intent discussed in chat 0e81ac8c; scaffolding implemented in ad41efed.",
    )

    # ---------- 19 one-click install bundle ----------
    add_paths(
        [
            "deploy/requirements-app.txt",
            "deploy/install.bat",
            "deploy/install.ps1",
            "deploy/prepare_install_bundle.ps1",
            "deploy/bundle_start.bat",
            "deploy/INSTALL_README.txt",
        ]
    )
    commit(
        "feat: add one-click install bundle for target Windows hosts",
        "prepare_install_bundle.ps1 builds SleepyDetect_Install; target runs "
        "install.bat to create venv, install dlib wheel + CUDA/CPU torch, migrate, "
        "and create admin. start via bundle_start.bat.\n\n"
        "Chat: ad41efed (CUDA migration packaging).",
    )

    # ---------- 20 catch-all remaining tracked files ----------
    git("add", "-A")
    # unstage ignored is automatic
    commit(
        "chore: add remaining project files for reconstructed history completeness",
        "Catch-all for any paths not covered by earlier thematic commits "
        "(keeps working tree clean after reconstruction).\n\n"
        "Source chats: afcf99e5, 1b48f9d0, 0e81ac8c, ad41efed.",
    )

    print("\n===== git log =====")
    git("log", "--oneline", "--decorate")
    print("\n===== status =====")
    git("status", "-sb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
