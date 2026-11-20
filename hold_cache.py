"""Measure and clear the hold latents written directly beside a project's rendered clips."""
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

try:
    from . import archive, holds
except ImportError:
    import archive
    import holds


class CacheConflict(ValueError):
    """The reviewed files changed or a render may be using them."""


def inventory(clips, referenced):
    groups = {"all": {}, "unused": {}}
    if clips.resolve() != clips:
        raise ValueError("Project storage cannot manage a linked project or clips folder.")
    try:
        entries = os.scandir(clips)
    except FileNotFoundError:
        return groups
    with entries:
        for entry in entries:
            if not entry.name.endswith(holds.HOLD_SUFFIX):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                continue
            stamp = (info.st_size, info.st_mtime_ns, info.st_ino)
            groups["all"][entry.name] = stamp
            if os.path.normcase(entry.name) not in referenced:
                groups["unused"][entry.name] = stamp
    return groups


def summary(files):
    snapshot = json.dumps(sorted(files.items())).encode()
    return {"files": len(files), "bytes": sum(stamp[0] for stamp in files.values()),
            "version": hashlib.sha256(snapshot).hexdigest()}


def project_cache(projects_dir, folder, data, scope=None, version=None):
    """A cleanup uses the exact set reviewed by the caller. Its report reflects any locked files left on disk."""
    rendering = bool(data.get("pending"))
    if scope is not None and rendering:
        raise CacheConflict("Wait for this project's queued and running renders to finish, then refresh storage.")
    projects = Path(projects_dir)
    clips = projects.resolve() / folder / "clips"
    relative = os.path.normcase(os.path.join(projects.name, folder, "clips"))
    referenced = {os.path.normcase(holds.hold_path(name)) for subfolder, name in archive.referenced_files(data)
                  if os.path.normcase(os.path.normpath(subfolder)) == relative}
    groups = inventory(clips, referenced)
    removed = {"files": 0, "bytes": 0}
    failed = 0
    if scope is not None:
        if scope not in groups:
            raise ValueError("Choose unused or all cached latents.")
        files = groups[scope]
        if version != summary(files)["version"]:
            raise CacheConflict("The cached files changed. Refresh storage and review the new sizes.")
        for name, (size, _, _) in files.items():
            try:
                (clips / name).unlink()
            except FileNotFoundError:
                continue
            except OSError:
                failed += 1
            else:
                removed["files"] += 1
                removed["bytes"] += size
        groups = inventory(clips, referenced)
    return {**{key: summary(files) for key, files in groups.items()},
            "free_bytes": shutil.disk_usage(clips if clips.is_dir() else clips.parent).free,
            "rendering": rendering, "removed": removed, "failed": failed}
