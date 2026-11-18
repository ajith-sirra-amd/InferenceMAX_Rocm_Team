"""Restore owned bind-mount permissions without rewriting unchanged inodes."""
import json
import os
from pathlib import Path
import stat
import sys
import time


def restore(root, uid, gid):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Expected a real mounted directory')
    started = time.monotonic()
    visited = changed = 0
    pending = [root]
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        visited += 1
        if (metadata.st_uid, metadata.st_gid) != (uid, gid):
            os.chown(path, uid, gid, follow_symlinks=False)
            changed += 1
        if stat.S_ISDIR(metadata.st_mode):
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)
    return dict(status='ownership_restored', root=str(root), visited=visited,
                changed=changed, seconds=time.monotonic()-started)


if __name__ == '__main__':
    root, uid, gid = sys.argv[1:]
    if root not in ('/logs', '/agentic-work'):
        raise ValueError('Unexpected bind mount')
    print(json.dumps(restore(root, int(uid), int(gid))), flush=True)
