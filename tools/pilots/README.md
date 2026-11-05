# Runtime per nuove simulazioni

Il ricalcolo delle metriche usa i record in `evidence/`. Per simulare di nuovo, preparare un runtime isolato su Linux. Nella disposizione usata dai worker, `project/` (checkout con `npm ci --prefix tools/showdown`) e `pilots/` (runtime) sono cartelle sorelle. Usare Foul Play al commit `8f541f80f91f960603673796559f796b10ccf0fb`.

1. Copiare in `pilots/` i file di questa cartella e clonare Foul Play in `pilots/foul-play/`.
2. Creare `pilots/.venv` per il pilota, installare le dipendenze di Foul Play (tra cui `poke-engine==0.0.48` e `websockets==14.1`). Creare `pilots/poke-venv` per il worker con `websockets==16.1.1` e `poke-env==0.16.1` per il pilota alternativo.
3. Dal checkout, creare un server per slot. Per il primo, usare `.venv/bin/python tools/pilots/prepare_seeded_server.py tools/showdown/node_modules/pokemon-showdown ../pilots/worker-server-0 18766`. Usare porte consecutive per gli altri slot.
4. Impostare `ICON_PILOT_DIR` al percorso assoluto del runtime. Gli script `run_final_battles.py` e `run_coverage_confirmation.py run` leggono i manifest. Simulare in una copia separata degli input congelati e con le porte libere.

Showdown registra gli input e verifica i seed. Il budget Foul Play è 25 ms e un thread. La ricerca nativa a tempo può variare fra esecuzioni.
