# brkraw-backup

CLI plugin that adds `brkraw backup` for scanning and archiving Paravision datasets.

## Install (editable)

```bash
pip install -e ./brkraw-backup
```

## Usage

```bash
brkraw backup init /path/to/raw_dir /path/to/archive_dir
brkraw backup info /path/to/archive_dir
brkraw backup scan  /path/to/raw_dir  /path/to/archive_dir
brkraw backup review /path/to/raw_dir /path/to/archive_dir
brkraw backup run   /path/to/raw_dir  /path/to/archive_dir
brkraw backup migrate /path/to/raw_dir /path/to/archive_dir --old-cache .brk-backup_cache
```

Notes:
- Archives are created as zip files under the archive directory.
- A JSON registry file is written to the archive directory (default: `.brkraw-backup-registry.json`).
- If you ran `brkraw backup init`, you can omit `raw_dir` / `archive_dir` for `scan|review|run|info|migrate` and it will use `config.yaml` values (`backup.rawdata`, `backup.archive`).
- If only one of `backup.rawdata` / `backup.archive` is set, commands will error and tell you which one is missing.
- Progress bars are shown on TTY when log level allows (disable with `--no-progress`).

## Deep integrity scan (optional)

Deep integrity scanning compares the raw directory file list against the archive ZIP file list.
It is intentionally optional because it can be slow on large datasets / network storage.

Options (available on `scan` and `run`):

- `--integrity off|new|all` (default: `off`)
  - `off`: skip deep checks
  - `new`: check only datasets where `last_backup` is newer than the last integrity check
  - `all`: check all raw+archive pairs (subject to limit)
- `--integrity-limit N` (default: `0`)
  - `0`: no limit
  - `N`: check at most N datasets per command invocation

The `info/scan/review/run` table includes an `INTEG` column:
- `-`: not checked
- `OK`: deep check passed
- `FAIL`: deep check found missing files
- `SKIP`: archive not a zip (cannot compare file lists)
