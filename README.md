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
