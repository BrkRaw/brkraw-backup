# brkraw-backup

CLI plugin that adds `brkraw backup` for scanning and archiving Bruker ParaVision datasets.

## Getting started

Install the plugin:

```bash
pip install -U git+ssh://git@github.com/BrkRaw/brkraw-backup.git
```

Initialize paths in `~/.brkraw/config.yaml`:

```bash
brkraw backup init /path/to/raw_dir /path/to/archive_dir
```

Scan and view status:

```bash
brkraw backup scan
brkraw backup info
```

Notes:
- Archives are stored under `backup.archive` as zip files.
- A JSON registry is written under the archive directory (default: `.brkraw-backup-registry.json`).
- If `backup.rawdata` / `backup.archive` are configured, most commands can omit positional paths.
- Progress bars show only on TTY; disable with `--no-progress`.
- Detailed `ISSUES:` sub-lines show only when log level is `DEBUG`.

## Commands

### `brkraw backup init`

Register raw/archive directories in `brkraw` config.

```bash
brkraw backup init /raw /archive
brkraw backup init /raw /archive --force
```

Writes:
- `backup.rawdata: /raw`
- `backup.archive: /archive`

### `brkraw backup scan`

Scan `rawdata` + `archive` and update the registry.

```bash
brkraw backup scan
brkraw backup scan /raw /archive
```

### `brkraw backup info`

Show the last recorded status from the JSON registry (no filesystem scan).

```bash
brkraw backup info
brkraw backup info /archive
brkraw backup info --status OK,ARCHIVED
brkraw backup info --status TODO
brkraw backup info --integ FAIL,WARN
brkraw backup info --exclude-status UNKNOWN
```

### `brkraw backup review`

Scan and show only non-OK datasets.

```bash
brkraw backup review
```

### `brkraw backup run`

Create/update archives for datasets that are missing a backup or mismatched.

```bash
brkraw backup run
brkraw backup run --only A,B,C
brkraw backup run --dry-run
brkraw backup run --rebuild
brkraw backup run --delete-raw --yes
```

Update behavior:
- Default: append missing files into existing zip (no full rebuild).
- `--rebuild`: recreate the zip from scratch.

### `brkraw backup migrate`

Import legacy `.brk-backup_cache` into the new JSON registry.

```bash
brkraw backup migrate /raw /archive
brkraw backup migrate /raw /archive --old-cache /archive/.brk-backup_cache
brkraw backup migrate /raw /archive --no-scan
```

### `brkraw backup about`

Show plugin + environment info (useful for debugging installs).

```bash
brkraw backup about
```

## Deep integrity scan (optional)

Deep integrity scanning compares:
- raw file list vs zip file list
- raw total bytes vs zip *uncompressed* bytes (`ZipInfo.file_size` sum)

This is optional because it can be slow on large datasets / network storage.

Options (available on `scan` and `run`):
- `--integrity off|new|all` (default: `off`)
  - `off`: skip deep checks
  - `new`: check only datasets that need a check (based on `last_backup` vs `integrity.checked_at`)
  - `all`: check all raw+archive pairs (subject to limit)
- `--integrity-limit N` (default: `0`)
  - `0`: no limit
  - `N`: check at most N datasets per command invocation

The status table includes an `INTEG` column:
- `-`: not checked
- `OK`: file list + byte totals match
- `WARN`: file list OK but byte totals differ
- `FAIL`: missing files found in archive
- `SKIP`: archive is not a zip (cannot compare file lists / bytes)
