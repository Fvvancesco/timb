"""Sequential persistent Showdown validation with explicit request deadlines."""
import json
from pathlib import Path
import selectors
import subprocess
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]


class TeamValidator:
    def __init__(self):
        self.process = None
        self.version = None

    def _read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if timeout <= 0 or not selector.select(timeout):
                raise subprocess.TimeoutExpired('Showdown validator', timeout)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Showdown validator closed its output')
        return json.loads(line)

    def start(self, timeout=30):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.process = subprocess.Popen(
            ['node', str(ROOT / 'tools/showdown/validate-team-stream.cjs')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        try:
            ready = self._read(timeout)
            if ready.get('ready') is not True:
                raise RuntimeError('Invalid validator handshake')
            self.version = ready['showdown_version']
        except BaseException:
            self.close()
            raise

    def validate(self, format, sets, timeout):
        started = perf_counter()
        if timeout <= 0:
            raise subprocess.TimeoutExpired('Showdown validator', timeout)
        try:
            self.start(timeout)
            self.process.stdin.write(json.dumps({'format': format, 'sets': sets}) + '\n')
            self.process.stdin.flush()
            result = self._read(timeout - (perf_counter() - started))
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result
        except BaseException:
            # A timed-out response must never be read as the next request's response.
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
