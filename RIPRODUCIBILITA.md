# Riproduzione degli esperimenti

Il [README](README.md) contiene installazione e demo. L’ambiente verificato usa Python 3.13.14, scikit-learn 1.9.1 e Showdown 0.11.11. I lockfile fissano le dipendenze principali.

## Verifica dai risultati salvati

Dopo l’installazione, da radice:

```bash
.venv/bin/python scripts/audit_release.py
.venv/bin/python scripts/audit_extensions.py
.venv/bin/python scripts/summarize_development.py
.venv/bin/python scripts/evaluate_final.py
.venv/bin/python scripts/analyze_final_statistics.py
.venv/bin/python scripts/summarize_solver_suite.py \
  --runtime reports/remote_solver/solver_runtime_distribution \
  --breadth reports/remote_solver/solver_breadth
```

Per le analisi della copertura, del pilota e delle repliche del predittore:

```bash
(cd evidence && sha256sum -c SHA256SUMS)
tar -xzf evidence/battle_verification.tar.gz
.venv/bin/python scripts/analyze_coverage_ablation.py
.venv/bin/python scripts/analyze_coverage_confirmation.py
.venv/bin/python scripts/analyze_matchup_panel.py
.venv/bin/python scripts/analyze_prediction_replication.py
.venv/bin/python scripts/analyze_prediction_replication_v2.py
```

L’archivio conserva specifiche, esiti e riscontri del controllo replay prodotti durante la raccolta. Gli analizzatori verificano integrità e completezza e ricalcolano le statistiche. Per ripetere il controllo del simulatore a partire dagli input grezzi di ciascuna battaglia servono anche i log completi dell’esecuzione.

## Riaddestramento e benchmark

Per ripetere la CV del nucleo e il benchmark del solver:

```bash
mkdir -p reports/reproduction
.venv/bin/python scripts/run_supervised.py --mode nested --kb-version v1 \
  --labels data/processed/dataset_v{2,3,4}/development_labels.jsonl \
  --output reports/reproduction/supervised_v1.json
.venv/bin/python scripts/run_solver_suite.py
```

Per la KB estesa passare `--kb-version v2`, per la rete aggiungere `--experiment neural`. `probe_tree_models.py --kb-version v1` e `--kb-version v2` confrontano albero e random forest. `select_development_model.py` accetta nuovi percorsi `--output` e `--report`. Lasciare il test finale escluso dalla scelta del modello.

Per la ricerca estesa usare una cartella nuova per i checkpoint:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/run_expanded_search.py prepare --output reports/reproduction/expanded
.venv/bin/python scripts/run_expanded_search.py run --output reports/reproduction/expanded --workers 8
```

Per la prova PyTorch, usare un ambiente con PyTorch 2.11.0+cu128 e una GPU CUDA, poi eseguire `scripts/run_torch_matchups.py` con `prepare`, `grid --gpu 0` e `finalize`, passando sempre lo stesso `--output` nuovo. Per il dettaglio del disegno vedere `config/torch_matchups_v1.json`. Le copie pubbliche di questi due runner rimuovono il vincolo sul nome del PC e generano gli hash del nuovo run. I sorgenti usati per i risultati pubblicati sono in `source_snapshot/` accanto ai rispettivi report.

## Completamento dell’avversario

| Esperimento | Comando dalla radice |
|---|---|
| Naive Bayes e Pokémon2Vec | `.venv/bin/python scripts/run_team_completion_extension.py evaluate` |
| Scenari di copertura con CP-SAT | `.venv/bin/python scripts/run_completion_solver_extension.py` |
| Profili e clustering | `.venv/bin/python scripts/run_team_profile_extension.py audit` e `evaluate --workers 4` |
| Profili sul campione esterno | `.venv/bin/python scripts/run_team_profiles_external.py evaluate` |

Per RDF2Vec installare `gensim==4.4.0 smart-open==9.0.2 wrapt==2.5.2` in un ambiente separato e usare `PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` con `scripts/run_rdf2vec_extension.py evaluate`. Gli split salvati permettono di partire dalle composizioni. Alcuni comandi aggiornano i report nella copia di lavoro.

## Fonti e simulazioni

I metadati `*.source.json` indicano gli URL e gli SHA-256 di Metamon Teams e del campione PokéChamp. Per rigenerare i dati umani serve PyArrow. Usare gli shard indicati sotto, verificando gli hash nei file di split e nei protocolli.

- [Metamon Teams](https://huggingface.co/datasets/jakegrigsby/metamon-teams), archivi Gen 9 OU `competitive` e `modern_replays`. `scripts/prepare_dataset.py` ricostruisce il catalogo da queste fonti in una copia di lavoro separata.
- [PokéChamp](https://huggingface.co/datasets/milkkarten/pokechamp), shard train 34, e [Metamon Raw Replays](https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays/tree/060367c26630e0d36308fd42673e5cf2f4b4421a/data), shard 30. `scripts/run_team_completion_extension.py prepare POKECHAMP.parquet METAMON30.parquet` prepara gli split originali.
- Shard Metamon 31 e 32 per i profili esterni e RDF2Vec, con URL e hash nei rispettivi file in `config/`. I due script usano `prepare PARQUET`. Per il test temporale, passare lo shard 30 e poi gli shard 33–36 a `scripts/run_team_completion_temporal.py`.

Le battaglie di ICON usano Foul Play al commit `8f541f80f91f960603673796559f796b10ccf0fb`, Showdown 0.11.11 e ricerca a tempo. [Istruzioni del pilota](tools/pilots/README.md). Versioni, seed e partizioni sono fissati. La ricerca a tempo può produrre traiettorie diverse su macchine diverse. Il riesame statistico parte dalle osservazioni salvate.
