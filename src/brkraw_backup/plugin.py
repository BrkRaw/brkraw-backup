from __future__ import annotations

import argparse
import logging
import sys
import time
import shutil
from pathlib import Path
from typing import Optional, TextIO, Any

from brkraw.core import config as config_core

from . import __version__
from .core import (
    DEFAULT_REGISTRY_NAME,
    archive_one,
    deep_integrity_check,
    load_registry,
    load_legacy_cache,
    mark_backup_result,
    migrate_legacy_cache_to_registry,
    maybe_delete_raw,
    render_scan_table,
    save_registry,
    scan_datasets,
    snapshots_from_registry,
    update_registry,
    verify_archive,
)

logger = logging.getLogger("brkraw")

_BANNER_PRINTED = False
_STDOUT: TextIO = sys.__stdout__
_STDERR: TextIO = sys.__stderr__


def _banner() -> None:
    global _BANNER_PRINTED
    if _BANNER_PRINTED:
        return
    _BANNER_PRINTED = True
    logger.info("brkraw-backup v%s", __version__)


def _make_progress(args: argparse.Namespace):
    def _pick_stream() -> TextIO:
        # Render progress to the same stream as the root logging handler when possible,
        # and clear the line before printing log output to avoid overwriting headers.
        root = logging.getLogger()
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                stream = getattr(handler, "stream", None)
                if stream is sys.stdout or stream is sys.__stdout__:
                    return _STDOUT
                if stream is sys.stderr or stream is sys.__stderr__:
                    return _STDERR
        return _STDERR

    stream = _pick_stream()
    enabled = (
        not bool(getattr(args, "no_progress", False))
        and stream.isatty()
        and logger.isEnabledFor(logging.INFO)
    )
    last_emit = 0.0
    last_line_len = 0
    start = time.time()

    def _label(step: str) -> str:
        if step.startswith("scan:"):
            return "scan"
        if step.startswith("zip:"):
            return "zip"
        return step.split(":", 1)[0] if ":" in step else step

    def reporter(current: int, total: int, step: str) -> None:
        nonlocal last_emit, last_line_len
        if total <= 0:
            return
        now = time.time()
        if now - last_emit < 0.1 and current < total:
            return
        last_emit = now

        label = _label(step)
        width = 24
        frac = min(1.0, max(0.0, current / total))
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(0.001, time.time() - start)
        rate = current / elapsed if current > 0 else 0.0
        remaining = max(0, total - current)
        eta = int(remaining / rate) if rate > 0 else -1
        eta_txt = f"{eta}s" if eta >= 0 else "?"
        line = f"{label} [{bar}] {current}/{total} ETA {eta_txt}"
        pad = " " * max(0, last_line_len - len(line))
        last_line_len = len(line)
        stream.write("\r" + line + pad)
        stream.flush()

    def done() -> None:
        if not enabled:
            return
        # Clear progress line to avoid leaving partial characters on screen.
        stream.write("\r" + (" " * last_line_len) + "\r\n")
        stream.flush()

    if not enabled:
        def reporter_noop(current: int, total: int, step: str) -> None:
            return

        def done_noop() -> None:
            return

        return reporter_noop, done_noop
    return reporter, done


def _get_backup_paths_from_config(*, root: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    cfg = config_core.resolve_config(root=root)
    backup_cfg = cfg.get("backup", {})
    if not isinstance(backup_cfg, dict):
        return None, None
    raw = backup_cfg.get("rawdata")
    arc = backup_cfg.get("archive")
    raw = raw.strip() if isinstance(raw, str) else None
    arc = arc.strip() if isinstance(arc, str) else None
    return raw or None, arc or None


def _configured_print_width(*, root: Optional[str]) -> Optional[int]:
    cfg = config_core.load_config(root=root)
    if not isinstance(cfg, dict):
        return None
    logging_cfg = cfg.get("logging", {})
    if not isinstance(logging_cfg, dict):
        return None
    value = logging_cfg.get("print_width")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _effective_print_width(*, root: Optional[str]) -> Optional[int]:
    configured = _configured_print_width(root=root)
    if configured:
        return configured
    if _STDERR.isatty():
        cols: int = shutil.get_terminal_size(fallback=(0, 0)).columns
        return cols if cols > 0 else None
    return None


def _resolve_paths(args: argparse.Namespace, *, need_raw: bool = True, need_archive: bool = True) -> tuple[Path, Path]:
    raw_cli = getattr(args, "raw_root", None) or getattr(args, "rawdata", None)
    arc_cli = getattr(args, "archive_root", None) or getattr(args, "archive", None)
    raw_cfg, arc_cfg = _get_backup_paths_from_config(root=getattr(args, "root", None))

    raw_value = raw_cli or raw_cfg
    arc_value = arc_cli or arc_cfg

    missing: list[str] = []
    if need_raw and not raw_value:
        missing.append("backup.rawdata (raw_root)")
    if need_archive and not arc_value:
        missing.append("backup.archive (archive_root)")
    if missing:
        hint = (
            "Provide the missing path(s) as CLI args (or --rawdata/--archive), "
            "or set them via: brkraw backup init <raw_root> <archive_root>."
        )
        raise ValueError(f"Missing required path(s): {', '.join(missing)}. {hint}")

    def _resolve(p: str) -> Path:
        # Use realpath-like resolution so symlinks are handled consistently.
        # strict=False allows resolution even if the directory doesn't exist yet.
        return Path(p).expanduser().resolve(strict=False)

    raw_path = _resolve(raw_value) if raw_value else Path()
    arc_path = _resolve(arc_value) if arc_value else Path()
    return raw_path, arc_path


def _maybe_prompt_save_backup_paths(
    args: argparse.Namespace,
    *,
    raw_root: Path,
    archive_root: Path,
) -> None:
    if getattr(args, "no_config_prompt", False):
        return
    if not sys.stdin.isatty():
        return

    raw_cli = getattr(args, "raw_root", None) or getattr(args, "rawdata", None)
    arc_cli = getattr(args, "archive_root", None) or getattr(args, "archive", None)
    if not raw_cli or not arc_cli:
        return

    config = config_core.load_config(root=getattr(args, "root", None))
    if config is None:
        return

    backup_cfg = config.get("backup")
    if not isinstance(backup_cfg, dict):
        backup_cfg = {}

    missing_keys: list[str] = []
    if not (isinstance(backup_cfg.get("rawdata"), str) and backup_cfg.get("rawdata", "").strip()):
        missing_keys.append("backup.rawdata")
    if not (isinstance(backup_cfg.get("archive"), str) and backup_cfg.get("archive", "").strip()):
        missing_keys.append("backup.archive")
    if not missing_keys:
        return

    raw_root = raw_root.resolve(strict=False)
    archive_root = archive_root.resolve(strict=False)

    prompt = (
        f"Config file exists but {', '.join(missing_keys)} not set.\n"
        f"Save current paths to config.yaml?\n"
        f"  rawdata : {raw_root}\n"
        f"  archive : {archive_root}\n"
        f"[y/N]: "
    )
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if answer not in {"y", "yes"}:
        return

    backup_cfg.setdefault("rawdata", str(raw_root))
    backup_cfg.setdefault("archive", str(archive_root))
    config["backup"] = backup_cfg
    config_core.write_config(config, root=getattr(args, "root", None))
    logger.info("Saved backup.rawdata/archive to config.yaml.")


def _add_init_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("raw_root", help="Directory containing raw datasets (subdirs).")
    parser.add_argument("archive_root", help="Directory to store dataset zip archives.")
    parser.add_argument(
        "--root",
        help="Override brkraw config root directory (default: BRKRAW_CONFIG_HOME or ~/.brkraw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config keys when present.",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("raw_root", nargs="?", help="Directory containing raw datasets (subdirs).")
    parser.add_argument("archive_root", nargs="?", help="Directory to store dataset zip archives.")
    parser.add_argument("--rawdata", dest="rawdata", help="Override config backup.rawdata for this command.")
    parser.add_argument("--archive", dest="archive", help="Override config backup.archive for this command.")
    parser.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY_NAME,
        help=f"Registry filename stored under archive_root (default: {DEFAULT_REGISTRY_NAME}).",
    )
    parser.add_argument(
        "--root",
        help="Override brkraw config root directory (default: BRKRAW_CONFIG_HOME or ~/.brkraw).",
    )
    parser.add_argument(
        "--no-config-prompt",
        action="store_true",
        help="Disable interactive prompt to save missing backup paths into config.yaml.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar rendering.",
    )
    parser.add_argument(
        "--integrity",
        choices=["off", "new", "all"],
        default="off",
        help="Optional deep integrity check (file list compare) for raw+archive pairs.",
    )
    parser.add_argument(
        "--integrity-limit",
        type=int,
        default=0,
        help="Max datasets to deep-check per run (0 means no limit).",
    )


def _maybe_run_integrity_checks(
    *,
    args: argparse.Namespace,
    registry: dict,
    snapshots: list,
) -> None:
    mode = getattr(args, "integrity", "off")
    if mode == "off":
        return
    limit = int(getattr(args, "integrity_limit", 0) or 0)
    checked = 0

    datasets = registry.get("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}
        registry["datasets"] = datasets

    reporter, done = _make_progress(args)
    candidates = []
    for snap in snapshots:
        if not getattr(snap, "raw_present", False) or not getattr(snap, "archive_present", False):
            continue
        if getattr(snap, "status", None) not in {"OK", "MISMATCH"}:
            continue
        if not snap.raw_path or not snap.archive_path:
            continue
        candidates.append(snap)

    total = len(candidates)
    for idx, snap in enumerate(candidates, start=1):
        reporter(idx, total, "integrity:pick")
        if limit and checked >= limit:
            break
        entry = datasets.get(snap.key, {})
        if not isinstance(entry, dict):
            entry = {}

        if mode == "new":
            last_backup = entry.get("last_backup")
            last_check = entry.get("integrity", {}).get("checked_at") if isinstance(entry.get("integrity"), dict) else None
            if isinstance(last_backup, str) and isinstance(last_check, str) and last_check >= last_backup:
                continue

        reporter(checked + 1, max(1, limit or total), "integrity:run")
        result = deep_integrity_check(Path(snap.raw_path), Path(snap.archive_path))
        entry["integrity"] = result
        datasets[snap.key] = entry
        checked += 1

    done()
    if checked:
        logger.info("Integrity checks completed: %d", checked)


def cmd_init(args: argparse.Namespace) -> int:
    _banner()
    raw_root = str(Path(args.raw_root).expanduser().resolve(strict=False))
    archive_root = str(Path(args.archive_root).expanduser().resolve(strict=False))

    config_core.ensure_initialized(root=args.root, create_config=True, exist_ok=True)
    config = config_core.load_config(root=args.root) or {}

    backup_cfg = config.get("backup")
    if not isinstance(backup_cfg, dict):
        backup_cfg = {}

    changed = False
    for key, value in (("rawdata", raw_root), ("archive", archive_root)):
        existing = backup_cfg.get(key)
        if isinstance(existing, str) and existing.strip() and not args.force:
            continue
        backup_cfg[key] = value
        changed = True

    config["backup"] = backup_cfg
    if changed:
        config_core.write_config(config, root=args.root)
        logger.info("Saved backup paths to config.yaml (section: backup).")
    else:
        logger.info("Config already has backup paths; skipped (use --force to overwrite).")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    _banner()
    try:
        raw_root, archive_root = _resolve_paths(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    registry_path = archive_root / args.registry

    _maybe_prompt_save_backup_paths(args, raw_root=raw_root, archive_root=archive_root)
    logger.debug("backup scan: raw_root=%s archive_root=%s registry=%s", raw_root, archive_root, registry_path)
    reporter, done = _make_progress(args)
    snapshots = scan_datasets(raw_root, archive_root, reporter=reporter)
    done()
    registry = load_registry(registry_path)
    width = _effective_print_width(root=args.root)
    logger.info("%s", render_scan_table(snapshots, max_width=width, registry=registry))
    _maybe_run_integrity_checks(args=args, registry=registry, snapshots=snapshots)
    registry = update_registry(registry, snapshots, raw_root=raw_root, archive_root=archive_root)
    save_registry(registry_path, registry)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    _banner()
    try:
        raw_root, archive_root = _resolve_paths(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    width = _effective_print_width(root=args.root)

    _maybe_prompt_save_backup_paths(args, raw_root=raw_root, archive_root=archive_root)
    logger.debug("backup review: raw_root=%s archive_root=%s", raw_root, archive_root)
    reporter, done = _make_progress(args)
    snapshots_all = scan_datasets(raw_root, archive_root, reporter=reporter)
    done()
    snapshots = [s for s in snapshots_all if s.status != "OK"]
    if not snapshots:
        logger.info("No issues found.")
        return 0
    registry = load_registry(archive_root / args.registry)
    logger.info("%s", render_scan_table(snapshots, max_width=width, registry=registry))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _banner()
    try:
        raw_root, archive_root = _resolve_paths(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    width = _effective_print_width(root=args.root)
    registry_path = archive_root / args.registry

    _maybe_prompt_save_backup_paths(args, raw_root=raw_root, archive_root=archive_root)
    logger.debug(
        "backup run: raw_root=%s archive_root=%s rebuild=%s dry_run=%s delete_raw=%s",
        raw_root,
        archive_root,
        bool(args.rebuild),
        bool(args.dry_run),
        bool(args.delete_raw),
    )
    reporter, done = _make_progress(args)
    snapshots = scan_datasets(raw_root, archive_root, reporter=reporter)
    done()
    registry = load_registry(registry_path)
    registry = update_registry(registry, snapshots, raw_root=raw_root, archive_root=archive_root)

    selected: Optional[set[str]] = None
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}

    todo = []
    for snap in snapshots:
        if selected is not None and snap.key not in selected:
            continue
        if snap.status in {"MISSING", "MISMATCH"}:
            if not snap.raw_present or not snap.raw_path:
                continue
            todo.append(Path(snap.raw_path).expanduser().resolve(strict=False))

    if not todo:
        logger.info("Nothing to archive.")
        save_registry(registry_path, registry)
        return 0

    reporter, done = _make_progress(args)
    for idx, raw_path in enumerate(todo, start=1):
        reporter(idx, len(todo), "archive:datasets")
        dest, added, skipped = archive_one(
            raw_path,
            archive_root,
            rebuild=args.rebuild,
            dry_run=args.dry_run,
            reporter=reporter,
        )
        ok = True
        if not args.dry_run:
            ok = verify_archive(dest)
        mark_backup_result(
            registry,
            key=raw_path.name,
            archive_path=dest,
            added=added,
            skipped=skipped,
            verified=ok,
            dry_run=bool(args.dry_run),
        )
        if ok:
            logger.info("Archived %s -> %s (added=%d skipped=%d)", raw_path.name, dest.name, added, skipped)
            maybe_delete_raw(raw_path, allow=args.delete_raw, confirmed=args.yes, dry_run=args.dry_run)
        else:
            logger.error("Archive verification failed: %s", dest)
    done()

    # Newly created/updated archives can be integrity-checked immediately.
    if getattr(args, "integrity", "off") != "off":
        snapshots = scan_datasets(raw_root, archive_root)
        _maybe_run_integrity_checks(args=args, registry=registry, snapshots=snapshots)

    reporter, done = _make_progress(args)
    snapshots = scan_datasets(raw_root, archive_root, reporter=reporter)
    done()
    registry = update_registry(registry, snapshots, raw_root=raw_root, archive_root=archive_root)
    save_registry(registry_path, registry)
    logger.info("%s", render_scan_table(snapshots, max_width=width, registry=registry))
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    _banner()
    try:
        raw_root, archive_root = _resolve_paths(args, need_raw=not bool(args.no_scan), need_archive=True)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    registry_path = archive_root / args.registry

    if not args.no_scan:
        _maybe_prompt_save_backup_paths(args, raw_root=raw_root, archive_root=archive_root)
    legacy_path = Path(args.old_cache).expanduser()
    if not legacy_path.is_absolute():
        legacy_path = archive_root / legacy_path

    if args.old_cache == ".brk-backup_cache":
        logger.info("Using default legacy cache path (relative to archive_root).")
    logger.info("Archive root: %s", archive_root)

    logger.debug(
        "backup migrate: legacy=%s registry=%s no_scan=%s overwrite=%s keep_logs=%s",
        legacy_path,
        registry_path,
        bool(args.no_scan),
        bool(args.overwrite),
        int(args.keep_logs),
    )
    logger.info("Command: brkraw backup migrate")
    logger.info("Migrating legacy cache -> registry")
    logger.info("Legacy cache: %s", legacy_path)
    logger.info("Registry: %s", registry_path)
    legacy = load_legacy_cache(legacy_path)
    if legacy is None:
        logger.error(
            "Legacy cache not found or unreadable: %s (tip: pass --old-cache /path/to/.brk-backup_cache)",
            legacy_path,
        )
        return 2

    registry = load_registry(registry_path)
    try:
        datasets = registry.get("datasets", {})
        existing = len(datasets) if isinstance(datasets, dict) else 0
    except Exception:
        existing = 0
    logger.info("Registry loaded: datasets=%d", existing)
    reporter, done = _make_progress(args)
    registry, migrated = migrate_legacy_cache_to_registry(
        legacy,
        registry,
        archive_root=archive_root,
        source_path=legacy_path,
        overwrite=bool(args.overwrite),
        keep_logs=int(args.keep_logs),
        reporter=reporter,
    )
    done()

    if not args.no_scan:
        reporter, done = _make_progress(args)
        snapshots = scan_datasets(raw_root, archive_root, reporter=reporter)
        done()
        registry = update_registry(registry, snapshots, raw_root=raw_root, archive_root=archive_root)
        width = _effective_print_width(root=args.root)
        logger.info("%s", render_scan_table(snapshots, max_width=width, registry=registry))

    save_registry(registry_path, registry)
    logger.info("Migrated %d dataset entries from %s", migrated, legacy_path.name)
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    _banner()
    try:
        _, archive_root = _resolve_paths(args, need_raw=False, need_archive=True)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    registry_path = archive_root / args.registry

    registry = load_registry(registry_path)
    snapshots = snapshots_from_registry(registry)
    if not snapshots:
        logger.info("Registry is empty: %s", registry_path)
        return 0
    width = _effective_print_width(root=args.root)
    logger.info("%s", render_scan_table(snapshots, max_width=width, registry=registry))
    return 0


def cmd_about(args: argparse.Namespace) -> int:
    _banner()
    logger.info("python: %s", sys.executable)
    logger.info("brkraw_backup: %s", __file__)
    cfg_root = config_core.resolve_root(getattr(args, "root", None))
    logger.info("config root: %s", cfg_root)
    raw_cfg, arc_cfg = _get_backup_paths_from_config(root=getattr(args, "root", None))
    logger.info("config backup.rawdata: %s", raw_cfg or "-")
    logger.info("config backup.archive: %s", arc_cfg or "-")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[name-defined]
    backup_parser = subparsers.add_parser(
        "backup",
        help="Archive raw datasets into a zip-based backup registry.",
    )
    sub = backup_parser.add_subparsers(dest="backup_command", metavar="command")

    init_p = sub.add_parser("init", help="Register raw/archive paths into brkraw config.yaml.")
    _add_init_args(init_p)
    init_p.set_defaults(func=cmd_init, parser=init_p)

    info_p = sub.add_parser("info", help="Show last scanned status from the JSON registry.")
    info_p.add_argument("archive_root", nargs="?", help="Directory that stores the JSON registry and archives.")
    info_p.add_argument("--archive", dest="archive", help="Override config backup.archive for this command.")
    info_p.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY_NAME,
        help=f"Registry filename stored under archive_root (default: {DEFAULT_REGISTRY_NAME}).",
    )
    info_p.add_argument(
        "--root",
        help="Override brkraw config root directory (default: BRKRAW_CONFIG_HOME or ~/.brkraw).",
    )
    info_p.set_defaults(func=cmd_registry, parser=info_p)

    # Backwards-compatible alias.
    reg_p = sub.add_parser("registry", help=argparse.SUPPRESS)
    reg_p.add_argument("archive_root", nargs="?")
    reg_p.add_argument("--archive", dest="archive")
    reg_p.add_argument("--registry", default=DEFAULT_REGISTRY_NAME)
    reg_p.add_argument("--root")
    reg_p.set_defaults(func=cmd_registry, parser=reg_p)

    about_p = sub.add_parser("about", help="Show plugin version and config paths.")
    about_p.add_argument(
        "--root",
        help="Override brkraw config root directory (default: BRKRAW_CONFIG_HOME or ~/.brkraw).",
    )
    about_p.set_defaults(func=cmd_about, parser=about_p)

    scan_p = sub.add_parser("scan", help="Scan raw/archive dirs and update registry.")
    _add_common_args(scan_p)
    scan_p.set_defaults(func=cmd_scan, parser=scan_p)

    review_p = sub.add_parser("review", help="Show only datasets with issues.")
    _add_common_args(review_p)
    review_p.set_defaults(func=cmd_review, parser=review_p)

    run_p = sub.add_parser("run", help="Create/update archives for missing/mismatched datasets.")
    _add_common_args(run_p)
    run_p.add_argument("--only", help="Comma-separated dataset names to process.")
    run_p.add_argument("--rebuild", action="store_true", help="Rebuild archives from scratch when present.")
    run_p.add_argument("--dry-run", action="store_true", help="Plan actions without writing.")
    run_p.add_argument("--delete-raw", action="store_true", help="Delete raw dataset after successful archive.")
    run_p.add_argument("--yes", action="store_true", help="Confirm destructive operations (e.g., --delete-raw).")
    run_p.set_defaults(func=cmd_run, parser=run_p)

    mig_p = sub.add_parser("migrate", help="Migrate legacy (.brk-backup_cache) into JSON registry.")
    _add_common_args(mig_p)
    mig_p.add_argument(
        "--old-cache",
        default=".brk-backup_cache",
        help="Legacy pickle cache filename or path (default: .brk-backup_cache under archive_root).",
    )
    mig_p.add_argument("--overwrite", action="store_true", help="Overwrite existing legacy_cache entries.")
    mig_p.add_argument("--keep-logs", type=int, default=50, help="Keep last N legacy log records (default: 50).")
    mig_p.add_argument("--no-scan", action="store_true", help="Skip a post-migration scan/update.")
    mig_p.set_defaults(func=cmd_migrate, parser=mig_p)

    backup_parser.set_defaults(
        func=lambda args: (args.parser.print_help() or 2),  # type: ignore[attr-defined]
        parser=backup_parser,
    )
