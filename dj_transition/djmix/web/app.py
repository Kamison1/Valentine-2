"""Small local Flask app: upload two songs, auto-merge, tweak, render."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from .. import DJMixError
from ..audio_io import SUPPORTED_EXTENSIONS, check_supported
from ..engine import TRANSITION_BAR_CHOICES, MixSession, MixSettings, waveform_peaks

MAX_UPLOAD_MB = 300


class _Job:
    """One uploaded pair of songs plus its render outputs."""

    def __init__(self, sid: str, folder: str, session: MixSession):
        self.sid, self.folder, self.session = sid, folder, session
        self.lock = threading.Lock()
        self.preview_count = 0
        self.overview = {
            "a": waveform_peaks(session.audio_a, 1400),
            "b": waveform_peaks(session.audio_b, 1400),
        }


def create_app(workdir: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    root = workdir or tempfile.mkdtemp(prefix="djmix_")
    os.makedirs(root, exist_ok=True)
    jobs: dict[str, _Job] = {}

    def get_job(sid: str) -> _Job:
        job = jobs.get(sid)
        if job is None:
            raise DJMixError("Session expired - please upload the songs again.")
        return job

    def preview_payload(job: _Job, settings: MixSettings) -> dict:
        s = job.session
        mix, plan, info, (stem_a, stem_b) = s.render_preview(settings, return_stems=True)
        job.preview_count += 1
        name = f"preview_{job.preview_count}.wav"
        for old in os.listdir(job.folder):
            if old.startswith("preview_"):
                os.unlink(os.path.join(job.folder, old))
        s.export(mix, os.path.join(job.folder, name), mp3=False)
        w0, w1 = info["preview_window"]
        info["rate_a"], info["rate_b"] = plan.rate_a, plan.rate_b
        return {
            "settings": settings.to_dict(),
            "info": info,
            "preview": {
                "url": f"/files/{job.sid}/{name}",
                "window": [w0, w1],
                "peaks": {"a": waveform_peaks(stem_a), "b": waveform_peaks(stem_b),
                          "mix": waveform_peaks(mix)},
                "beats": s.beat_markers(plan, w0, w1),
            },
        }

    def render_full(job: _Job, settings: MixSettings) -> dict:
        mix, _, info = job.session.render_full(settings)
        paths = job.session.export(mix, os.path.join(job.folder, "merged.wav"))
        return {kind: f"/files/{job.sid}/{os.path.basename(p)}?v={uuid.uuid4().hex[:6]}"
                for kind, p in paths.items()} | {"render_seconds": info["render_seconds"],
                                                 "duration": info["total_duration"]}

    @app.errorhandler(DJMixError)
    def _user_error(exc):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(413)
    def _too_big(_exc):
        return jsonify({"error": f"Upload too large (limit {MAX_UPLOAD_MB} MB)."}), 413

    @app.get("/")
    def index():
        return render_template("index.html", bar_choices=TRANSITION_BAR_CHOICES,
                               extensions=",".join(SUPPORTED_EXTENSIONS))

    @app.post("/api/automerge")
    def automerge():
        files = [request.files.get("song_a"), request.files.get("song_b")]
        if not all(f and f.filename for f in files):
            raise DJMixError("Please choose both Song A and Song B.")
        sid = uuid.uuid4().hex[:12]
        folder = os.path.join(root, sid)
        os.makedirs(folder)
        paths = []
        try:
            for label, f in zip("ab", files):
                name = secure_filename(f.filename) or f"song_{label}"
                check_supported(name)
                path = os.path.join(folder, f"{label}_{name}")
                f.save(path)
                paths.append(path)
            session = MixSession(*paths, log=app.logger.info)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        job = jobs[sid] = _Job(sid, folder, session)

        with job.lock:
            settings = session.auto_settings()
            payload = preview_payload(job, settings)
            payload["full"] = render_full(job, settings)
        a, b = session.analysis_a, session.analysis_b
        payload.update({
            "sid": sid,
            "analysis": {"a": a.summary(), "b": b.summary()},
            "downbeats_a": [round(float(t), 3) for t in a.downbeat_times],
            "suggestions": [c.as_dict() for c in session.suggestions(settings)],
            "overview": job.overview,
        })
        return jsonify(payload)

    @app.post("/api/preview")
    def preview():
        body = request.get_json(force=True)
        job = get_job(body.get("sid", ""))
        settings = MixSettings.from_dict(body.get("settings", {}))
        with job.lock:
            payload = preview_payload(job, settings)
        payload["suggestions"] = [c.as_dict() for c in job.session.suggestions(settings)]
        return jsonify(payload)

    @app.post("/api/render")
    def render():
        body = request.get_json(force=True)
        job = get_job(body.get("sid", ""))
        settings = MixSettings.from_dict(body.get("settings", {}))
        with job.lock:
            return jsonify({"full": render_full(job, settings)})

    @app.get("/files/<sid>/<path:name>")
    def files(sid, name):
        job = jobs.get(sid)
        if job is None:
            abort(404)
        return send_from_directory(job.folder, name, as_attachment=request.args.get("dl") == "1")

    return app
