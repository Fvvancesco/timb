# ICON · Costruzione e riparazione spiegabile di squadre Pokémon

ICON completa e ripara squadre Gen 9 OU per Pokémon Showdown. La KB RDF/OWL ricava capacità dei set e le mosse che le giustificano. CP-SAT applica i vincoli e cerca una riparazione minima. Il progetto valuta separatamente copertura, previsione della vittoria, scelta dei candidati e completamento dell’avversario.

- [Riproduzione](RIPRODUCIBILITA.md) per i comandi e le fonti degli input.
- [Valutazione del predittore e dei selettori](reports/VALUTAZIONE_FINALE.md), [benchmark dei solver](reports/solver_comparison.md) e [completamento dell’avversario](reports/team_completion_v1/RESULTS.md).

## Avvio rapido

Dalla radice del progetto con Python 3.13, uv e Node.js:

```bash
uv sync --frozen --python 3.13
npm ci --prefix tools/showdown
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_kb_enrichment.py
```

Il demo materializza le inferenze sul catalogo e produce una riparazione validata da Showdown. Le spiegazioni mostrano le mosse che soddisfano i requisiti.

| Cartella | Contenuto |
|---|---|
| `src/icon_pokemon/` e `ontology/` | Implementazione, schema OWL e annotazioni degli effetti |
| `scripts/` e `config/` | Procedure sperimentali, analisi, griglie, manifest e seed |
| `tools/` e `tests/` | Collegamento a Showdown e al pilota, controlli di correttezza |
| `data/processed/`, `models/` e `reports/` | Input congelati, predizioni, modello e risultati |
| `evidence/` | Record necessari a ricalcolare le analisi dei lotti in battaglia |
