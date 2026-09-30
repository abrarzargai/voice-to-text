"""Runs Whisper in a separate process so a transcription can be stopped instantly.

faster-whisper only yields between 30-second windows (and can retry a hard window for a long
time), so a stop flag alone can take minutes to be noticed. Killing the worker process is
immediate. The process stays alive between jobs to keep loaded models cached; it is only
restarted after a stop or a crash.
"""
import multiprocessing as mp


class WorkerError(Exception):
    """A failure reported by the worker process (message already includes the error type)."""


def _serve(conn):
    from faster_whisper import WhisperModel

    models = {}
    while True:
        try:
            path, opts = conn.recv()
        except EOFError:
            return
        try:
            key = (opts["model"], opts["device"])
            if key not in models:
                compute_type = "int8" if opts["device"] == "cpu" else "default"
                models[key] = WhisperModel(opts["model"], device=opts["device"], compute_type=compute_type)
            conn.send(("loaded",))
            segments, info = models[key].transcribe(
                path,
                language=opts["language"],
                task=opts["task"],
                beam_size=opts["beam_size"],
                vad_filter=opts["vad"],
                condition_on_previous_text=False,
                # With VAD the silence is already cut out. Whisper's own "no speech" check on top of it
                # can then drop every segment of unclear speech (seen with "base" on Urdu voice notes).
                **({"no_speech_threshold": None} if opts["vad"] else {}),
            )
            conn.send(("info", info.language, info.language_probability, info.duration))
            for seg in segments:
                conn.send(("segment", seg.start, seg.end, seg.text))
            conn.send(("done",))
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}"))


class WhisperWorker:
    def __init__(self):
        self.proc = self.conn = None

    def _ensure(self):
        if self.proc is None or not self.proc.is_alive():
            ctx = mp.get_context("spawn")  # never fork a threaded server
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_serve, args=(child,), daemon=True, name="whisper-worker")
            self.proc.start()
            child.close()

    def kill(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.join(5)
            self.conn.close()
        self.proc = self.conn = None

    def run(self, path, opts, check_stop, poll=0.3):
        """Yield ("loaded",), ("info", lang, prob, duration) and ("segment", start, end, text).

        check_stop() is called while waiting and should raise to stop; the worker is then killed.
        """
        self._ensure()
        self.conn.send((str(path), opts))
        finished = False
        try:
            while True:
                check_stop()
                if not self.conn.poll(poll):
                    if not self.proc.is_alive():
                        raise WorkerError("The Whisper process exited unexpectedly (out of memory?)")
                    continue
                msg = self.conn.recv()
                if msg[0] == "error":
                    finished = True
                    raise WorkerError(msg[1])
                if msg[0] == "done":
                    finished = True
                    return
                yield msg
        finally:
            if not finished:  # stopped, crashed or abandoned: the worker's state is unknown
                self.kill()
