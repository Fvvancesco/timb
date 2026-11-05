# Confronto dei costruttori e distribuzioni del tempo al successo

900 run su tre istanze (100 seed per metodo), più 300 run su venti istanze (cinque seed per metodo). Budget 10 secondi; stessa macchina `Francesco-Desktop`, esecuzione sequenziale, un thread CP-SAT, KB e validatore già caricati. Gli insuccessi restano nei denominatori.

La soglia di qualità è fissata dal pilot prima dei run: confronta lessicograficamente membri conservati, copertura offensiva, opzioni difensive statiche e ridondanza. Solo le soglie con certificato sono chiamate ottime.

| Istanza | Catalogo | Metodo | Valida entro 10 s | Soglia entro 10 s | Mediana tempo alla soglia | p90 tempo alla soglia | Sequenze distinte |
|---|---:|---|---:|---:|---:|---:|---:|
| instance-01 (repair) | 24 | random | 100/100 | 100/100 | 0.0152 s | 0.0420 s | 99 |
| instance-01 (repair) | 24 | greedy | 100/100 | 100/100 | 0.0016 s | 0.0018 s | 2 |
| instance-01 (repair) | 24 | cp_sat | 100/100 | 100/100 | 0.0058 s | 0.0081 s | 10 |
| instance-04 (completion) | 160 | random | 100/100 | 100/100 | 0.0047 s | 0.0165 s | 100 |
| instance-04 (completion) | 160 | greedy | 100/100 | 100/100 | 0.0048 s | 0.0050 s | 100 |
| instance-04 (completion) | 160 | cp_sat | 100/100 | 100/100 | 0.0090 s | 0.0093 s | 90 |
| instance-07 (repair) | 594 | random | 100/100 | 100/100 | 0.4666 s | 1.4938 s | 100 |
| instance-07 (repair) | 594 | greedy | 100/100 | 100/100 | 0.0172 s | 0.0179 s | 6 |
| instance-07 (repair) | 594 | cp_sat | 100/100 | 100/100 | 0.0578 s | 0.0744 s | 87 |

## Verifica su tutte le venti richieste

| Metodo | Valida entro 10 s | Soglia entro 10 s | Certificati entro 10 s |
|---|---:|---:|---:|
| random | 100/100 | 100/100 | non applicabile |
| greedy | 100/100 | 100/100 | non applicabile |
| cp_sat | 100/100 | 100/100 | 100/100 |

Ogni istanza ha lo stesso numero di repliche: il riepilogo dà peso uguale alle istanze. Le curve complete per ogni metodo/istanza sono nel JSON, insieme agli stati di arresto. I percentili usano tutti i run: se meno del 90% raggiunge la soglia, il p90 è non raggiunto entro il limite.

Il numero di sequenze distinte esclude i timestamp e confronta le squadre e qualità degli incumbent. Una sola sequenza CP-SAT osservata significa che i seed non hanno prodotto percorsi osservabili diversi; le cento ripetizioni descrivono soprattutto variabilità di cronometraggio. Non basta cambiare il seed per dimostrare cento ricerche indipendenti.

Preparazione e warm-up sono riportati nei `summary.json` sorgente e restano fuori dal tempo della singola richiesta. Il pilot con validatore a freddo e il lotto locale interrotto non sono mescolati a questi tempi.

Riepilogo riproducibile: `.venv/bin/python scripts/summarize_solver_suite.py --runtime CARTELLA_RUNTIME --breadth CARTELLA_BREADTH`. Il comando rifiuta run mancanti, duplicati, host o codice differenti.
