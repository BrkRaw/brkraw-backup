from __future__ import annotations

from dataclasses import dataclass, asdict
import datetime as _dt
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import zipfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Callable, Set

from brkraw.core.formatter import format_data
from brkraw.dataclasses.study import Study

logger = logging.getLogger("brkraw")


DEFAULT_REGISTRY_NAME = ".brkraw-backup-registry.json"
ProgressReporter = Callable[[int, int, str], None]


@dataclass(frozen=True)
class DatasetSnapshot:
    key: str
    raw_path: Optional[str]
    archive_path: Optional[str]
    raw_present: bool
    archive_present: bool
    raw_valid: bool
    archive_valid: bool
    raw_scan_count: Optional[int]
    archive_scan_count: Optional[int]
    raw_sw_version: Optional[str]
    archive_sw_version: Optional[str]
    raw_bytes: Optional[int]
    archive_bytes: Optional[int]
    issues: Tuple[str, ...]
    status: str


def _utcnow() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, filenames in os.walk(path):
        for name in filenames:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _zip_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _is_candidate_raw_dir(name: str) -> bool:
    if not name or name.startswith("."):
        return False
    if "import" in name:
        return False
    return True


def _iter_raw_datasets(raw_root: Path) -> Dict[str, Path]:
    datasets: Dict[str, Path] = {}
    if not raw_root.exists():
        return datasets
    for entry in sorted(raw_root.iterdir()):
        if not entry.is_dir():
            continue
        if not _is_candidate_raw_dir(entry.name):
            continue
        datasets[entry.name] = entry
    return datasets


def _iter_archive_files(archive_root: Path) -> Dict[str, Path]:
    datasets: Dict[str, Path] = {}
    if not archive_root.exists():
        return datasets
    for entry in sorted(archive_root.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        lower = name.lower()
        if lower.endswith(".zip"):
            key = name[:-4]
        elif lower.endswith(".pvdatasets"):
            key = name[: -len(".pvdatasets")]
        elif name.endswith("PvDatasets"):
            key = name[: -len("PvDatasets")]
            if key.endswith("."):
                key = key[:-1]
        else:
            continue

        key = key.strip()
        if not key:
            continue
        # NOTE: Do not call zipfile.is_zipfile() here. It can be expensive on
        # network-mounted archives. Validation happens later when we load metadata.
        datasets[key] = entry
    return datasets


def _archive_key(zip_path: Path) -> str:
    # Retained for compatibility; avoid opening archives here.
    name = zip_path.name
    lower = name.lower()
    if lower.endswith(".zip"):
        return name[:-4]
    if lower.endswith(".pvdatasets"):
        return name[: -len(".pvdatasets")]
    if name.endswith("PvDatasets"):
        key = name[: -len("PvDatasets")]
        return key[:-1] if key.endswith(".") else key
    return zip_path.stem


def _load_loader(path: Path) -> Tuple[bool, Optional[int], Optional[str]]:
    try:
        study = Study.from_path(path)
    except Exception:
        return False, None, None
    try:
        scan_count = len(study.avail)
    except Exception:
        scan_count = None
    # Avoid BrukerLoader sw_version parsing here to keep scans lightweight and
    # prevent deep rule selection logs during simple status scans.
    return True, scan_count, None


def scan_datasets(
    raw_root: Path,
    archive_root: Path,
    *,
    reporter: Optional[ProgressReporter] = None,
) -> List[DatasetSnapshot]:
    logger.info("Scan start: raw_root=%s archive_root=%s", raw_root, archive_root)
    raw_datasets = _iter_raw_datasets(raw_root)
    archive_datasets = _iter_archive_files(archive_root)
    logger.info("Discovered candidates: raw=%d archive=%d", len(raw_datasets), len(archive_datasets))

    keys = sorted(set(raw_datasets) | set(archive_datasets))
    snapshots: List[DatasetSnapshot] = []

    total = len(keys)
    for idx, key in enumerate(keys, start=1):
        if reporter:
            reporter(idx, total, "scan:datasets")
        raw_path = raw_datasets.get(key)
        arc_path = archive_datasets.get(key)

        raw_present = raw_path is not None and raw_path.exists()
        arc_present = arc_path is not None and arc_path.exists()

        raw_valid, raw_scans, raw_ver = (False, None, None)
        if raw_present and raw_path is not None:
            raw_valid, raw_scans, raw_ver = _load_loader(raw_path)

        arc_valid, arc_scans, arc_ver = (False, None, None)
        if arc_present and arc_path is not None:
            arc_valid, arc_scans, arc_ver = _load_loader(arc_path)

        issues: List[str] = []
        if raw_present and not raw_valid:
            issues.append("raw_invalid")
        if arc_present and not arc_valid:
            issues.append("archive_corrupt")
        if raw_present and not arc_present:
            issues.append("archive_missing")
        # raw_missing is expected once data is archived; treat it as an issue
        # only when the archive is not readable/valid.
        if not raw_present and arc_present and not arc_valid:
            issues.append("raw_missing")
        if not raw_present and not arc_present:
            issues.append("both_missing")
        if raw_present and arc_present and raw_valid and arc_valid:
            if raw_scans is not None and arc_scans is not None and raw_scans != arc_scans:
                issues.append("scan_count_mismatch")
            if raw_ver and arc_ver and raw_ver != arc_ver:
                issues.append("paravision_version_mismatch")

        status = _derive_status(raw_present, arc_present, raw_valid, arc_valid, issues)
        if issues:
            logger.debug("Dataset issues: %s -> %s", key, ",".join(issues))
        snapshots.append(
            DatasetSnapshot(
                key=key,
                raw_path=str(raw_path.resolve(strict=False)) if raw_path else None,
                archive_path=str(arc_path.resolve(strict=False)) if arc_path else None,
                raw_present=raw_present,
                archive_present=arc_present,
                raw_valid=raw_valid,
                archive_valid=arc_valid,
                raw_scan_count=_safe_int(raw_scans),
                archive_scan_count=_safe_int(arc_scans),
                raw_sw_version=raw_ver,
                archive_sw_version=arc_ver,
                raw_bytes=_dir_size_bytes(raw_path) if raw_present and raw_path else None,
                archive_bytes=_zip_size_bytes(arc_path) if arc_present and arc_path else None,
                issues=tuple(issues),
                status=status,
            )
        )

    if reporter:
        reporter(total, total, "scan:done")
    logger.info("Scan done: datasets=%d", len(snapshots))
    return snapshots


def _derive_status(
    raw_present: bool,
    archive_present: bool,
    raw_valid: bool,
    archive_valid: bool,
    issues: Sequence[str],
) -> str:
    if archive_present and not archive_valid:
        return "CORRUPT"
    if raw_present and not raw_valid:
        return "INVALID"
    if raw_present and not archive_present:
        return "MISSING"
    if not raw_present and archive_present and archive_valid:
        return "ARCHIVED"
    if not raw_present and archive_present:
        return "RAW_REMOVED"
    if any(i in {"scan_count_mismatch", "paravision_version_mismatch"} for i in issues):
        return "MISMATCH"
    if raw_present and archive_present and raw_valid and archive_valid:
        return "OK"
    return "UNKNOWN"


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "created_at": _utcnow(), "updated_at": _utcnow(), "datasets": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("registry must be a JSON object")
        data.setdefault("datasets", {})
        return data
    except Exception as exc:
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, backup)
            logger.warning("Registry unreadable (%s); backed up to %s", path, backup)
        except Exception:
            logger.warning("Registry unreadable (%s): %s", path, exc)
            pass
        return {"version": 1, "created_at": _utcnow(), "updated_at": _utcnow(), "datasets": {}}


def save_registry(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = dict(data)
    payload["updated_at"] = _utcnow()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)
    logger.debug("Saved registry: %s", path)


def update_registry(
    registry: Dict[str, Any],
    snapshots: Iterable[DatasetSnapshot],
    *,
    raw_root: Path,
    archive_root: Path,
) -> Dict[str, Any]:
    data = dict(registry)
    data.setdefault("version", 1)
    data.setdefault("created_at", _utcnow())
    data.setdefault("datasets", {})
    data["raw_root"] = str(raw_root.resolve(strict=False))
    data["archive_root"] = str(archive_root.resolve(strict=False))
    datasets = dict(data["datasets"])
    now = _utcnow()
    for snap in snapshots:
        entry = datasets.get(snap.key, {})
        if not isinstance(entry, dict):
            entry = {}
        entry.update(asdict(snap))
        entry["last_scan"] = now
        datasets[snap.key] = entry
    data["datasets"] = datasets
    logger.debug("Updated registry entries: %d", len(datasets))
    return data


def mark_backup_result(
    registry: Dict[str, Any],
    *,
    key: str,
    archive_path: Path,
    added: int,
    skipped: int,
    verified: bool,
    dry_run: bool,
) -> None:
    datasets = registry.setdefault("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}
        registry["datasets"] = datasets
    entry = datasets.get(key, {})
    if not isinstance(entry, dict):
        entry = {}
    entry.update(
        {
            "last_backup": _utcnow(),
            "last_backup_archive_path": str(archive_path),
            "last_backup_added": int(added),
            "last_backup_skipped": int(skipped),
            "last_backup_verified": bool(verified),
            "last_backup_dry_run": bool(dry_run),
        }
    )
    datasets[key] = entry
    logger.debug(
        "Backup result: key=%s archive=%s added=%d skipped=%d verified=%s dry_run=%s",
        key,
        archive_path,
        added,
        skipped,
        verified,
        dry_run,
    )


def _status_cell(status: str) -> Mapping[str, Any]:
    if status == "OK":
        return {"value": status, "color": "green", "bold": True}
    if status == "ARCHIVED":
        return {"value": status, "color": "green", "bold": True}
    if status in {"MISSING", "CORRUPT", "INVALID"}:
        return {"value": status, "color": "red", "bold": True}
    if status == "MISMATCH":
        return {"value": status, "color": "yellow", "bold": True}
    if status == "RAW_REMOVED":
        return {"value": status, "color": "cyan"}
    return {"value": status, "color": "gray"}


def _zip_root_prefix(zf: zipfile.ZipFile) -> str:
    names = [n.strip("/") for n in zf.namelist() if n.strip("/")]
    if not names:
        return ""
    first = names[0].split("/")[0]
    for n in names[1:]:
        if not n.startswith(first + "/") and n != first:
            return ""
    return first


def _zip_file_set(path: Path) -> Set[str]:
    with zipfile.ZipFile(path, "r") as zf:
        prefix = _zip_root_prefix(zf)
        files: Set[str] = set()
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            name = name.strip("/")
            if prefix and name.startswith(prefix + "/"):
                name = name[len(prefix) + 1 :]
            if name:
                files.add(name)
        return files


def _dir_file_set(path: Path) -> Set[str]:
    files: Set[str] = set()
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            full = Path(dirpath) / fname
            try:
                rel = full.relative_to(path).as_posix()
            except Exception:
                continue
            if rel:
                files.add(rel)
    return files


def deep_integrity_check(raw_path: Path, archive_path: Path) -> Dict[str, Any]:
    """Optional heavy check: compare raw vs archive file lists (zip-only)."""
    raw_path = raw_path.resolve(strict=False)
    archive_path = archive_path.resolve(strict=False)
    started = _dt.datetime.now(tz=_dt.timezone.utc)

    result: Dict[str, Any] = {
        "checked_at": started.isoformat(),
        "method": "filelist",
        "raw_path": str(raw_path),
        "archive_path": str(archive_path),
    }

    if not raw_path.exists() or not raw_path.is_dir():
        result.update({"ok": False, "error": "raw_not_found"})
        return result
    if not archive_path.exists() or not archive_path.is_file():
        result.update({"ok": False, "error": "archive_not_found"})
        return result
    if not zipfile.is_zipfile(archive_path):
        result.update({"ok": None, "error": "archive_not_zip"})
        return result

    try:
        raw_files = _dir_file_set(raw_path)
        arc_files = _zip_file_set(archive_path)
    except Exception as exc:
        result.update({"ok": False, "error": f"exception:{type(exc).__name__}"})
        return result

    missing = sorted(raw_files - arc_files)
    extra = sorted(arc_files - raw_files)
    result.update(
        {
            "ok": len(missing) == 0,
            "raw_files": len(raw_files),
            "archive_files": len(arc_files),
            "missing_files": len(missing),
            "extra_files": len(extra),
            "missing_examples": missing[:20],
            "extra_examples": extra[:20],
        }
    )
    return result


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "-"
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "?"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if units[idx] in {"B", "KB"}:
        return f"{int(size)}{units[idx]}"
    return f"{size:.1f}{units[idx]}"


def _format_backup_time(value: Optional[str]) -> str:
    if not value:
        return "-"
    try:
        dt = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return str(value)
    # Render in local time for operator friendliness.
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 1:
        return text[:max_len]
    return text[: max_len - 1] + "…"


def render_scan_table(
    snapshots: Sequence[DatasetSnapshot],
    *,
    max_width: Optional[int] = None,
    registry: Optional[Mapping[str, Any]] = None,
) -> str:
    reg_datasets: Mapping[str, Any] = {}
    if registry and isinstance(registry.get("datasets"), Mapping):
        reg_datasets = registry.get("datasets", {})  # type: ignore[assignment]

    rows: List[Dict[str, Any]] = []
    max_key_len = max((len(s.key) for s in snapshots), default=len("DATASET"))
    key_w = max_key_len + 3

    def _last_backup(key: str) -> str:
        entry = reg_datasets.get(key)
        if isinstance(entry, Mapping):
            return _format_backup_time(entry.get("last_backup"))  # type: ignore[arg-type]
        return "-"

    for snap in snapshots:
        raw_scans = "-" if not snap.raw_present else (snap.raw_scan_count if snap.raw_scan_count is not None else "?")
        arc_scans = "-" if not snap.archive_present else (
            snap.archive_scan_count if snap.archive_scan_count is not None else "?"
        )
        issues = ",".join(snap.issues) if snap.issues else ""
        raw_sz = _format_bytes(snap.raw_bytes) if snap.raw_present else "-"
        arc_sz = _format_bytes(snap.archive_bytes) if snap.archive_present else "-"
        bkp_at = _last_backup(snap.key)
        rows.append(
            {
                "key": {"value": snap.key, "bold": True},
                "rawn": raw_scans,
                "arcn": arc_scans,
                "rawsz": raw_sz,
                "arcsz": arc_sz,
                "bkp": bkp_at,
                "status": _status_cell(snap.status),
                "issues": issues,
            }
        )

    # fixed-width columns (not including key/issues)
    raw_w = max(3, len("RAW"))
    arc_w = max(3, len("ARC"))
    rawsz_w = max(6, len("RAW_SZ"))
    arcz_w = max(6, len("ARC_SZ"))
    bkp_w = len("BACKUP_AT")
    status_w = max(10, len("STATUS"))
    gap = "  "

    fixed = (
        raw_w
        + len(gap)
        + arc_w
        + len(gap)
        + rawsz_w
        + len(gap)
        + arcz_w
        + len(gap)
        + bkp_w
        + len(gap)
        + status_w
        + len(gap)
    )

    issues_w: Optional[int] = None
    if max_width is not None:
        # Ensure at least some room for key.
        min_key = 20
        # Prefer user's "max+3" key width, but clamp to max_width.
        max_key_allowed = max(min_key, max_width - fixed)
        key_w = min(key_w, max_key_allowed)
        remaining = max_width - fixed - key_w
        issues_w = max(0, remaining)

        # truncate content to fit without wrapping
        for row in rows:
            key_text = str(row.get("key", ""))
            if isinstance(row.get("key"), Mapping):
                key_text = str(row["key"].get("value", ""))  # type: ignore[assignment]
                row["key"] = dict(row["key"])  # shallow copy
                row["key"]["value"] = _truncate(key_text, key_w)  # type: ignore[index]
            row["issues"] = _truncate(str(row.get("issues", "")), issues_w)

    template = (
        f"{{key: <{key_w}}}"
        f"{{rawn: >{raw_w}}}{gap}"
        f"{{arcn: >{arc_w}}}{gap}"
        f"{{rawsz: >{rawsz_w}}}{gap}"
        f"{{arcsz: >{arcz_w}}}{gap}"
        f"{{bkp: <{bkp_w}}}{gap}"
        f"{{status: <{status_w}}}{gap}"
        f"{{issues}}"
    )
    header = (
        f"{'DATASET': <{key_w}}"
        f"{'RAW': >{raw_w}}{gap}"
        f"{'ARC': >{arc_w}}{gap}"
        f"{'RAW_SZ': >{rawsz_w}}{gap}"
        f"{'ARC_SZ': >{arcz_w}}{gap}"
        f"{'BACKUP_AT': <{bkp_w}}{gap}"
        f"{'STATUS': <{status_w}}{gap}"
        f"ISSUES"
    )
    sep = "-" * len(header)
    body = format_data(rows, template, width=None, on_missing="placeholder")
    return "\n".join([header, sep, body]) if body else "\n".join([header, sep])


def archive_one(
    raw_path: Path,
    archive_root: Path,
    *,
    rebuild: bool,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[Path, int, int]:
    key = raw_path.name
    dest = archive_root / f"{key}.zip"
    logger.debug("Archiving dataset: %s -> %s (rebuild=%s dry_run=%s)", raw_path, dest, rebuild, dry_run)
    if rebuild or not dest.exists():
        added = _write_zip_from_dir(raw_path, dest, root_name=key, dry_run=dry_run, reporter=reporter)
        return dest, added, 0

    root_name = _archive_key(dest) or key
    added, skipped = _append_missing_files(raw_path, dest, root_name=root_name, dry_run=dry_run, reporter=reporter)
    return dest, added, skipped


def _walk_files(root: Path) -> Iterable[Tuple[Path, str]]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            yield full, rel


def _write_zip_from_dir(
    root: Path,
    dest: Path,
    *,
    root_name: str,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
) -> int:
    files = list(_walk_files(root))
    total = len(files)
    if dry_run:
        if reporter:
            reporter(total, total, "zip:plan")
        return len(files)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (full, rel) in enumerate(files, start=1):
            if reporter:
                reporter(idx, total, "zip:write")
            arcname = f"{root_name}/{rel}".strip("/")
            zf.write(full, arcname)
    if reporter:
        reporter(total, total, "zip:done")
    return len(files)


def _append_missing_files(
    root: Path,
    dest: Path,
    *,
    root_name: str,
    dry_run: bool,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[int, int]:
    if not dest.exists():
        raise FileNotFoundError(dest)
    existing: set[str] = set()
    with zipfile.ZipFile(dest, "r") as zf:
        for info in zf.infolist():
            name = info.filename.rstrip("/")
            if name:
                existing.add(name)

    add: List[Tuple[Path, str]] = []
    skipped = 0
    files = list(_walk_files(root))
    total = len(files)
    for idx, (full, rel) in enumerate(files, start=1):
        if reporter:
            reporter(idx, total, "zip:diff")
        arcname = f"{root_name}/{rel}".strip("/")
        if arcname in existing:
            skipped += 1
            continue
        add.append((full, arcname))

    if dry_run:
        if reporter:
            reporter(total, total, "zip:plan")
        return len(add), skipped

    with zipfile.ZipFile(dest, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        total_add = len(add)
        for idx, (full, arcname) in enumerate(add, start=1):
            if reporter:
                reporter(idx, total_add, "zip:append")
            zf.write(full, arcname)
    if reporter:
        reporter(total, total, "zip:done")
    return len(add), skipped


def maybe_delete_raw(
    raw_path: Path,
    *,
    allow: bool,
    confirmed: bool,
    dry_run: bool,
) -> None:
    if not allow:
        return
    if not confirmed:
        raise ValueError("Refusing to delete raw data without --yes.")
    if dry_run:
        logger.debug("Dry-run delete raw dataset: %s", raw_path)
        return
    logger.warning("Deleting raw dataset: %s", raw_path)
    shutil.rmtree(raw_path)


def verify_archive(zip_path: Path) -> bool:
    ok, _, _ = _load_loader(zip_path)
    return ok


def snapshots_from_registry(registry: Mapping[str, Any]) -> List[DatasetSnapshot]:
    datasets = registry.get("datasets", {})
    if not isinstance(datasets, Mapping):
        return []
    snapshots: List[DatasetSnapshot] = []
    for key in sorted(datasets.keys()):
        entry = datasets.get(key)
        if not isinstance(entry, Mapping):
            continue

        issues_value = entry.get("issues", ())
        if isinstance(issues_value, (list, tuple)):
            issues = tuple(str(x) for x in issues_value)
        elif issues_value is None:
            issues = ()
        else:
            issues = (str(issues_value),)

        def _abs_path(value: object) -> Optional[str]:
            if not isinstance(value, str) or not value.strip():
                return None
            try:
                return str(Path(value).expanduser().resolve(strict=False))
            except Exception:
                return value

        snapshots.append(
            DatasetSnapshot(
                key=str(entry.get("key", key)),
                raw_path=_abs_path(entry.get("raw_path")),
                archive_path=_abs_path(entry.get("archive_path")),
                raw_present=bool(entry.get("raw_present", False)),
                archive_present=bool(entry.get("archive_present", False)),
                raw_valid=bool(entry.get("raw_valid", False)),
                archive_valid=bool(entry.get("archive_valid", False)),
                raw_scan_count=_safe_int(entry.get("raw_scan_count")),
                archive_scan_count=_safe_int(entry.get("archive_scan_count")),
                raw_sw_version=entry.get("raw_sw_version"),
                archive_sw_version=entry.get("archive_sw_version"),
                raw_bytes=_safe_int(entry.get("raw_bytes")),
                archive_bytes=_safe_int(entry.get("archive_bytes")),
                issues=issues,
                status=str(entry.get("status", "UNKNOWN")),
            )
        )
    return snapshots


class _LegacyPlaceholder:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D401
        # The legacy cache pickles objects from brkraw 0.3.x (e.g. brkraw.lib.backup.NamedTuple).
        # We don't import those modules; instead, hydrate them into a generic placeholder.
        return type(f"Legacy_{module.replace('.', '_')}_{name}", (_LegacyPlaceholder,), {})


def load_legacy_cache(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return _LegacyUnpickler(f).load()
    except Exception:
        return None


def migrate_legacy_cache_to_registry(
    legacy_cache: object,
    registry: Dict[str, Any],
    *,
    archive_root: Path,
    source_path: Path,
    overwrite: bool = False,
    keep_logs: int = 50,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[Dict[str, Any], int]:
    try:
        raw_n = len(getattr(legacy_cache, "raw_data", []) or [])
        arc_n = len(getattr(legacy_cache, "arc_data", []) or [])
    except Exception:
        raw_n, arc_n = -1, -1
    logger.info("Legacy cache summary: raw_entries=%s archive_entries=%s", raw_n, arc_n)
    datasets = registry.setdefault("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}
        registry["datasets"] = datasets

    raw_data = getattr(legacy_cache, "raw_data", []) or []
    arc_data = getattr(legacy_cache, "arc_data", []) or []
    log_data = getattr(legacy_cache, "log_data", []) or []

    by_pid: Dict[int, Dict[str, Any]] = {}
    for raw in raw_data:
        try:
            pid = int(getattr(raw, "data_pid", -1))
        except Exception:
            continue
        by_pid.setdefault(pid, {})
        by_pid[pid]["raw"] = {
            "path": getattr(raw, "path", None),
            "garbage": getattr(raw, "garbage", None),
            "removed": getattr(raw, "removed", None),
            "backup": getattr(raw, "backup", None),
        }

    for arc in arc_data:
        try:
            pid = int(getattr(arc, "data_pid", -1))
        except Exception:
            continue
        by_pid.setdefault(pid, {})
        arcs = by_pid[pid].setdefault("archives", [])
        arc_fname = getattr(arc, "path", None)
        arcs.append(
            {
                "filename": arc_fname,
                "path": str((archive_root / arc_fname)) if arc_fname else None,
                "garbage": getattr(arc, "garbage", None),
                "crashed": getattr(arc, "crashed", None),
                "issued": getattr(arc, "issued", None),
            }
        )

    migrated = 0
    items = [item for _, item in sorted(by_pid.items(), key=lambda kv: kv[0])]
    total = len(items)
    if reporter and total:
        reporter(0, total, "migrate:datasets")
    for idx, item in enumerate(items, start=1):
        if reporter:
            reporter(idx, total, "migrate:datasets")
        raw = item.get("raw") or {}
        key = raw.get("path")
        if not isinstance(key, str) or not key.strip():
            continue
        key = key.strip()

        entry = datasets.get(key, {})
        if not isinstance(entry, dict):
            entry = {}

        if "legacy_cache" in entry and not overwrite:
            continue

        legacy_payload: Dict[str, Any] = {
            "source": str(source_path),
            "migrated_at": _utcnow(),
            "raw": raw,
            "archives": item.get("archives", []),
        }
        if keep_logs:
            tail = log_data[-keep_logs:] if isinstance(log_data, list) else []
            legacy_payload["logs_tail"] = [
                {
                    "datetime": getattr(rec, "datetime", None),
                    "method": getattr(rec, "method", None),
                    "message": getattr(rec, "message", None),
                }
                for rec in tail
            ]

        entry["legacy_cache"] = legacy_payload
        entry.setdefault("key", key)
        datasets[key] = entry
        migrated += 1

    registry["datasets"] = datasets
    if reporter and total:
        reporter(total, total, "migrate:done")
    registry.setdefault("migrations", [])
    migrations = registry["migrations"]
    if isinstance(migrations, list):
        migrations.append(
            {
                "type": "legacy_pickle_cache",
                "source": str(source_path),
                "migrated_at": _utcnow(),
                "datasets_migrated": migrated,
            }
        )
    return registry, migrated
