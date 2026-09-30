"""Settings and records, persisted as JSON in data/db.json. Audio files live in data/audio/.

A record is one recording plus the history of everything produced from it:

    {"id", "name", "created", "updated",
     "audio": {"file", "name", "size"} or None,
     "transcripts": [version, ...], "minutes": [version, ...]}

A version is {"id", "n", "created", "text", ...metadata}. "n" counts up per record and kind and
is never reused, so "v3" keeps meaning the same thing after v2 is deleted.
"""
import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

KINDS = ("transcripts", "minutes")


def new_id():
    return uuid.uuid4().hex[:12]


class Store:
    def __init__(self, data_dir, default_settings):
        self.dir = Path(data_dir)
        self.audio_dir = self.dir / "audio"
        self.path = self.dir / "db.json"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.fresh = not self.path.exists()
        data = {} if self.fresh else json.loads(self.path.read_text(encoding="utf-8"))
        saved = data.get("settings", {})
        settings = {**default_settings, **saved}
        settings["options"] = {**default_settings["options"], **saved.get("options", {})}
        self.data = {"version": 1, "settings": settings, "records": data.get("records", [])}
        if self.fresh:
            self._save()

    def _save(self):
        """Write atomically so a crash never leaves a half-written database. Caller holds the lock."""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---------- settings ----------
    def settings(self):
        with self.lock:
            return copy.deepcopy(self.data["settings"])

    def update_settings(self, patch):
        with self.lock:
            options = patch.pop("options", None)
            self.data["settings"].update(patch)
            if options:
                self.data["settings"]["options"].update(options)
            self._save()
            return copy.deepcopy(self.data["settings"])

    # ---------- records ----------
    def _find(self, rid):
        return next((r for r in self.data["records"] if r["id"] == rid), None)

    def get(self, rid):
        with self.lock:
            rec = self._find(rid)
            return copy.deepcopy(rec) if rec else None

    def name_of(self, rid):
        with self.lock:
            rec = self._find(rid)
            return rec["name"] if rec else None

    def summaries(self):
        with self.lock:
            out = []
            for r in self.data["records"]:
                out.append({
                    "id": r["id"], "name": r["name"], "created": r["created"], "updated": r["updated"],
                    "has_audio": bool(r.get("audio")), "audio_name": (r.get("audio") or {}).get("name"),
                    "transcripts": len(r["transcripts"]), "minutes": len(r["minutes"]),
                })
            return sorted(out, key=lambda r: -r["updated"])

    def unique_name(self, name, exclude=None):
        """The name itself if free, else "name (2)", "name (3)", ... (case-insensitive)."""
        base = re.sub(r"\s+", " ", name or "").strip()[:120] or "Untitled recording"
        with self.lock:
            taken = {r["name"].casefold() for r in self.data["records"] if r["id"] != exclude}
        candidate, n = base, 2
        while candidate.casefold() in taken:
            candidate, n = f"{base} ({n})", n + 1
        return candidate

    def create(self, name, audio=None, created=None):
        now = created or time.time()
        with self.lock:
            rec = {"id": new_id(), "name": self.unique_name(name), "created": now, "updated": now,
                   "audio": audio, "transcripts": [], "minutes": []}
            self.data["records"].append(rec)
            self._save()
            return copy.deepcopy(rec)

    def rename(self, rid, name):
        name = re.sub(r"\s+", " ", name or "").strip()[:120]
        if not name:
            raise ValueError("The name can't be empty")
        with self.lock:
            rec = self._find(rid)
            if rec is None:
                raise KeyError(rid)
            if self.unique_name(name, exclude=rid) != name:
                raise ValueError(f'Another record is already named "{name}"')
            rec["name"] = name
            rec["updated"] = time.time()
            self._save()
            return rec["name"]

    def delete(self, rid):
        """Remove a record; returns its audio file name (for the caller to delete) or None."""
        with self.lock:
            rec = self._find(rid)
            if rec is None:
                raise KeyError(rid)
            self.data["records"].remove(rec)
            self._save()
            return (rec.get("audio") or {}).get("file")

    # ---------- versions ----------
    def add_version(self, rid, kind, fields, created=None):
        """Append a new version; returns it, or None if the record was deleted meanwhile."""
        assert kind in KINDS
        with self.lock:
            rec = self._find(rid)
            if rec is None:
                return None
            n = max((v["n"] for v in rec[kind]), default=0) + 1
            rec.setdefault("next_n", {})
            n = max(n, rec["next_n"].get(kind, 1))
            rec["next_n"][kind] = n + 1  # never reuse a number, even after deleting the newest
            version = {"id": new_id(), "n": n, "created": created or time.time(), **fields}
            rec[kind].append(version)
            rec["updated"] = max(rec["updated"], version["created"])
            self._save()
            return copy.deepcopy(version)

    def find_version(self, rid, vid):
        """(kind, version copy) or (None, None)."""
        with self.lock:
            rec = self._find(rid)
            for kind in KINDS:
                for v in (rec or {}).get(kind, []):
                    if v["id"] == vid:
                        return kind, copy.deepcopy(v)
        return None, None

    def edit_version(self, rid, vid, text):
        with self.lock:
            rec = self._find(rid)
            version = next((v for kind in KINDS for v in (rec or {}).get(kind, []) if v["id"] == vid), None)
            if version is None:
                raise KeyError(vid)
            version["text"] = text
            version["edited"] = time.time()
            rec["updated"] = version["edited"]
            self._save()
            return copy.deepcopy(version)

    def delete_version(self, rid, vid):
        with self.lock:
            rec = self._find(rid)
            for kind in KINDS:
                for v in (rec or {}).get(kind, []):
                    if v["id"] == vid:
                        rec[kind].remove(v)
                        self._save()
                        return kind
        raise KeyError(vid)

    # ---------- one-time import of the old outputs/ folder ----------
    def import_outputs(self, outputs_dir):
        """Turn transcripts (and their "- minutes" files) from outputs/ into records. Files are kept."""
        outputs_dir = Path(outputs_dir)
        if not outputs_dir.is_dir():
            return 0
        files = sorted((p for p in outputs_dir.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
        minutes_re = re.compile(r"^(.*) - minutes(?: \(\d+\))?$")
        transcripts = [p for p in files if p.suffix.lower() in (".txt", ".srt")]
        minutes = [p for p in files if p.suffix.lower() == ".md"]
        count = 0
        for p in transcripts:
            text = p.read_text(encoding="utf-8", errors="replace")
            first = next((line for line in text.splitlines() if line.strip()), "")
            fmt = "srt" if p.suffix.lower() == ".srt" else "timestamps" if re.match(r"^\[\d", first) else "txt"
            rec = self.create(p.stem, created=p.stat().st_mtime)
            v = self.add_version(rec["id"], "transcripts",
                                 {"text": text, "format": fmt, "imported": p.name}, created=p.stat().st_mtime)
            for m in [m for m in minutes if (mm := minutes_re.match(m.stem)) and mm.group(1) == p.stem]:
                self.add_version(rec["id"], "minutes",
                                 {"text": m.read_text(encoding="utf-8", errors="replace"), "source": v["id"],
                                  "source_n": v["n"], "imported": m.name}, created=m.stat().st_mtime)
                minutes.remove(m)
            count += 1
        for m in minutes:  # minutes whose transcript no longer exists
            rec = self.create(minutes_re.sub(r"\1", m.stem), created=m.stat().st_mtime)
            self.add_version(rec["id"], "minutes", {"text": m.read_text(encoding="utf-8", errors="replace"),
                                                     "imported": m.name}, created=m.stat().st_mtime)
            count += 1
        return count
