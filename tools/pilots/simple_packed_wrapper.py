"""Run one seeded SimpleHeuristicsPlayer client with the frozen packed team."""
import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import random

random.seed(int(os.environ['ICON_PYTHON_SEED']))
Path(os.environ['ICON_PYTHON_STATE_FILE']).write_text(
    hashlib.sha256(repr(random.getstate()).encode()).hexdigest() + '\n')

from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder


class PackedTeam(Teambuilder):
    def yield_team(self):
        return Path(os.environ['ICON_PACKED_TEAM']).read_text().strip()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--websocket-uri', required=True)
    parser.add_argument('--ps-username', required=True)
    parser.add_argument('--pokemon-format', required=True)
    parser.add_argument('--bot-mode', choices=('accept_challenge','challenge_user'), required=True)
    parser.add_argument('--user-to-challenge')
    args, _ = parser.parse_known_args()
    player = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(args.ps_username, None),
        server_configuration=ServerConfiguration(args.websocket_uri, 'https://play.pokemonshowdown.com/action.php?'),
        battle_format=args.pokemon_format, team=PackedTeam())
    if args.bot_mode == 'accept_challenge':
        async def logged_in():
            await player.ps_client.logged_in.wait()
        future = asyncio.run_coroutine_threadsafe(logged_in(), player.ps_client.loop)
        await asyncio.wait_for(asyncio.wrap_future(future), timeout=20)
        print(f'Waiting for a {args.pokemon_format} challenge', flush=True)
        await player.accept_challenges(None, 1)
    else:
        await player.send_challenges(args.user_to_challenge, 1)
    # The parent records the final battle and terminates the clients.
    await asyncio.sleep(140)


if __name__ == '__main__':
    asyncio.run(main())
