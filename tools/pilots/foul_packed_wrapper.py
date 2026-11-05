"""Keep upstream pilot behavior but use the catalog's validated packed team."""
import hashlib, os, random, runpy
import asyncio
from pathlib import Path
from urllib.parse import urlsplit
random.seed(int(os.environ['ICON_PYTHON_SEED']))
Path(os.environ['ICON_PYTHON_STATE_FILE']).write_text(hashlib.sha256(repr(random.getstate()).encode()).hexdigest()+'\n')
import importlib
loader = importlib.import_module("fp.teams.load_team")
original = loader.load_team
def load_team(name):
    _, team_dict, filename = original(name)
    return Path(os.environ['ICON_PACKED_TEAM']).read_text().strip(), team_dict, filename
loader.load_team = load_team

from fp.websocket_client import PSWebsocketClient, logger

original_login = PSWebsocketClient.login

async def login(self):
    address = urlsplit(self.address)
    if (self.password is not None or address.scheme != 'ws'
            or address.hostname != '127.0.0.1' or address.port not in range(18766, 18780)):
        return await original_login(self)
    await self.get_id_and_challstr()
    await self.send_message("", [f"/trn {self.username},0,"])
    await asyncio.sleep(3)
    logger.info("Successfully logged in as a local guest")
    return self.username

PSWebsocketClient.login = login
runpy.run_path('run.py',run_name='__main__')
