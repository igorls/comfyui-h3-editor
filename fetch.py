"""Downloads the model files a fresh install needs, for the editor's Get the models panel.

One download runs at a time, in a thread of its own, one file after another. Each file is written
to its name with .part added, in the folder it belongs in, and renamed to its own name only once
its size and sha256 match what its repository publishes. A .part left by a stopped download is
carried on from where it stopped. A file already in place is never written to or replaced.

The plan comes from the server's own list (project.SETUP_FILES); nothing here takes a URL from a
request, and only https URLs on Hugging Face are fetched, including where they redirect to.

Hot reloads of the pack leave this module as it is (it is not in dev_reload.MODULES), so a download
in progress keeps reporting to the same state.
"""

import hashlib
import http.client
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

PART = ".part"
CHUNK = 1 << 20
TIMEOUT = 60
ATTEMPTS = 5
HOST = "huggingface.co"
AGENT = "h3-editor-setup"

_lock = threading.Lock()
_job = None


class Stopped(Exception):
    """The user stopped the download."""


class IncompleteDownload(Exception):
    """The connection closed before the whole file came."""


class Refused(ValueError):
    """A plan or a URL this module will not fetch."""


def allowed(url):
    """Whether url is one this module fetches: https on Hugging Face."""
    parts = urlsplit(url)
    return parts.scheme == "https" and parts.hostname == HOST and not parts.username and not parts.password


class HttpsOnly(urllib.request.HTTPRedirectHandler):
    """Follow redirects to https only: Hugging Face sends large files on to its CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise urllib.error.HTTPError(newurl, code, "refused a redirect off https", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def opener():
    return urllib.request.build_opener(HttpsOnly)


def check_plan(plan):
    """Refuse a plan whose URL is not on Hugging Face, or whose file is missing a size, a hash or a place."""
    for item in plan:
        if not allowed(item.get("url") or ""):
            raise Refused("only files on %s are downloaded, not %s" % (HOST, item.get("url")))
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0 or len(item.get("sha256") or "") != 64:
            raise Refused("%s has no size or sha256 to check it against" % item.get("id"))
        if not os.path.isabs(item.get("dest") or ""):
            raise Refused("%s has no folder to go in" % item.get("id"))


def status():
    """The download as the editor shows it, or None before the first one."""
    with _lock:
        if _job is None:
            return None
        files = [dict(row) for row in _job["files"]]
        job = {key: _job[key] for key in ("state", "error", "started", "finished")}
    total = sum(row["bytes"] for row in files)
    done = sum(row["done"] for row in files)
    job.update(files=files, bytes=total, done=done, rate=_rate(files))
    return job


def _rate(files):
    """Bytes a second over the last few seconds, for the file being written."""
    for row in files:
        if row["state"] == "downloading" and row.get("rate"):
            return row["rate"]
    return 0


def running():
    with _lock:
        return _job is not None and _job["state"] == "running"


def start(plan, finished=None):
    """Download plan in a thread: a list of {id, url, dest, bytes, sha256}. Returns the status.

    finished(item) is called after each file is in place, such as to have ComfyUI list it.
    """
    global _job
    check_plan(plan)
    with _lock:
        if _job is not None and _job["state"] == "running":
            raise Refused("a download is already running")
        stop = threading.Event()
        _job = {"state": "running", "error": None, "started": time.time(), "finished": None, "stop": stop,
                "files": [{"id": item["id"], "file": os.path.basename(item["dest"]), "bytes": item["bytes"], "done": 0,
                           "state": "waiting", "error": None} for item in plan]}
        job = _job
    threading.Thread(target=_run, args=(job, plan, finished), name="h3-editor-setup", daemon=True).start()
    return status()


def cancel():
    """Stop the download after the piece being written. What was written stays, to carry on from."""
    with _lock:
        if _job is not None and _job["state"] == "running":
            _job["stop"].set()
            return True
    return False


def _update(job, index, **values):
    with _lock:
        job["files"][index].update(values)


def _run(job, plan, finished):
    state, error = "done", None
    for index, item in enumerate(plan):
        if job["stop"].is_set():
            state = "stopped"
            break
        try:
            placed = fetch(item, job["stop"], lambda **values: _update(job, index, **values))
            _update(job, index, state="done" if placed else "skipped", done=item["bytes"], rate=0)
            if placed and finished:
                try:
                    finished(item)
                except Exception:
                    pass
        except Stopped:
            _update(job, index, state="stopped", rate=0)
            state = "stopped"
            break
        except Exception as failure:
            _update(job, index, state="failed", error=str(failure), rate=0)
            state, error = "failed", "%s: %s" % (os.path.basename(item["dest"]), failure)
            break
    with _lock:
        job.update(state=state, error=error, finished=time.time())
        for row in job["files"]:
            if row["state"] == "waiting" and state != "done":
                row["state"] = "not started"


def fetch(item, stop=None, report=None):
    """Download one file of a plan to its place. Returns True when it was placed, False when a file was already there.

    Retries a broken connection, carrying on from what was written. report(**values) is told
    state, done (bytes) and rate as it goes.
    """
    report = report or (lambda **values: None)
    stop = stop or threading.Event()
    dest, size = item["dest"], item["bytes"]
    if not allowed(item["url"]):
        raise Refused("only files on %s are downloaded" % HOST)
    if os.path.lexists(dest):
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + PART
    last = None
    for attempt in range(ATTEMPTS):
        try:
            digest = _download(item["url"], part, size, stop, report)
            break
        except (urllib.error.URLError, http.client.HTTPException, socket.timeout, ConnectionError, TimeoutError, IncompleteDownload) as failure:
            if isinstance(failure, urllib.error.HTTPError) and failure.code < 500 and failure.code not in (408, 429):
                raise RuntimeError("the server answered %s %s" % (failure.code, failure.reason))
            last = failure
            report(state="retrying", rate=0)
            if stop.wait(min(30, 2 * (attempt + 1))):
                raise Stopped()
    else:
        raise RuntimeError("the connection kept breaking (%s). Download again to carry on from here" % last)
    if digest != item["sha256"].lower():
        os.remove(part)
        raise RuntimeError("it came out different from the file its repository publishes, so it was deleted. Download again")
    return place(part, dest)


def _download(url, part, size, stop, report):
    """Write url to part, carrying on from what part holds. Returns the sha256 of the whole file."""
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    digest = hashlib.sha256()
    if have:
        report(state="checking", done=have)
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(CHUNK), b""):
                if stop.is_set():
                    raise Stopped()
                digest.update(block)
    if have == size:
        return digest.hexdigest()
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, **({"Range": "bytes=%d-" % have} if have else {})})
    report(state="downloading", done=have)
    try:
        response = opener().open(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as failure:
        if failure.code == 416 and have:
            # The server has nothing past what part holds: start the file again.
            os.remove(part)
            raise IncompleteDownload("the part already written did not fit the file, so it starts again")
        raise
    with response:
        mode = "ab"
        if have and response.status != 206:
            # The server sends the whole file: write it from the start.
            have, digest, mode = 0, hashlib.sha256(), "wb"
        elif have and not (response.headers.get("Content-Range") or "").startswith("bytes %d-" % have):
            raise IncompleteDownload("the server sent another part of the file than asked for")
        mark, marked = time.monotonic(), have
        with open(part, mode) as f:
            while True:
                if stop.is_set():
                    raise Stopped()
                block = response.read(CHUNK)
                if not block:
                    break
                if have + len(block) > size:
                    f.close()
                    os.remove(part)
                    raise RuntimeError("the server sent more than the file's published size")
                f.write(block)
                digest.update(block)
                have += len(block)
                now = time.monotonic()
                if now - mark >= 1:
                    report(state="downloading", done=have, rate=int((have - marked) / (now - mark)))
                    mark, marked = now, have
    report(done=have)
    if have < size:
        raise IncompleteDownload("%d of %d bytes came" % (have, size))
    return digest.hexdigest()


def place(part, dest):
    """Give part its own name, without replacing a file already there. Returns False, and drops part, if one is."""
    try:
        # A hard link fails when dest exists, so a file that appeared meanwhile is never replaced.
        os.link(part, dest)
    except FileExistsError:
        os.remove(part)
        return False
    except (OSError, AttributeError, NotImplementedError):
        # A filesystem without hard links. Windows' rename refuses an existing dest too.
        if os.path.lexists(dest):
            os.remove(part)
            return False
        os.rename(part, dest)
        return True
    os.remove(part)
    return True
