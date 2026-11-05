# Ricerca estesa degli iperparametri sullo sviluppo

Approfondimento richiesto dopo la revisione del nucleo. Serve a verificare quanto i risultati dipendano dalla griglia ristretta e dall'augmentation; non garantisce un miglioramento. I risultati del test finale già aperto non vengono usati per scegliere configurazioni né riscritti.

## Dati e partizioni

Si usano gli stessi 5.463 esiti decisivi di sviluppo v2–v4 del confronto precedente. V5 non viene aggiunto a questo confronto. Le squadre del test finale e i relativi esiti sono esclusi. Prima del fit si verificano manifest, provenienza delle etichette e hash del codice.

CV annidata a tre fold esterni e tre interni, ripetuta con seed 11, 23 e 47. Appartenenza delle squadre ai fold identica al nucleo. Una partita entra in una partizione solo se entrambe le squadre vi appartengono; gli incontri che attraversano il confine sono esclusi e contabilizzati. Gli iperparametri sono scelti sulla log loss media nei fold interni; il fold esterno valuta la configurazione scelta senza partecipare alla selezione.

## Griglie cartesiane

| Famiglia | Iperparametro | Valori |
|---|---|---|
| Logistica | C | 0,001; 0,01; 0,1; 1; 10 |
| Boosting | Profondità | 1; 2; 3 |
| Boosting | Numero alberi | 40; 80; 160 |
| Boosting | Learning rate | 0,03; 0,1; 0,3 |
| Boosting | Minimo esempi per foglia | 5; 10; 20 |
| MLP | Strati nascosti | 32; 64; 64→32 |
| MLP | Alpha L2 | 0,01; 1; 10 |
| MLP | Learning rate iniziale | 0,0003; 0,001; 0,003 |
| MLP | Batch size | 32; 64; 128 |

Cinque configurazioni logistiche, 81 boosting e 81 MLP. La MLP mantiene Adam, ReLU, massimo 300 epoche, tolleranza 10⁻⁴ e pazienza 20 sulla loss di training; niente split casuale per early stopping. Gli avvisi di convergenza sono conservati. I seed sono repliche, non iperparametri fra cui scegliere il migliore.

La ricerca si ripete per KB v1/v2 e con/senza augmentation, per **18.036 fit interni** e fino a 432 fit aggiuntivi per valutazioni esterne e curve. I refit esterni coincidenti tra griglia ristretta ed estesa vengono riusati.

## Controllo dell'augmentation

Lo scambio `(A,B,y) → (B,A,1−y)` avviene soltanto dopo la separazione dei fold. Le due orientazioni pesano 0,5 ciascuna. Logistica e MLP usano uno StandardScaler comune alle due posizioni, stimato esclusivamente sulle righe di training prima dell'augmentation. Così la normalizzazione commuta con lo scambio e resta la stessa nelle due condizioni. Gli alberi non vengono standardizzati.

Le probabilità usate per selezione e valutazione restano simmetrizzate in entrambe le condizioni. Si salvano anche le metriche delle probabilità grezze per diagnosticare l'effetto della simmetrizzazione. Ogni condizione sceglie i propri iperparametri nei fold interni.

Per MLP, raddoppiare le righe aumenta il numero di mini-batch per epoca. Pesi, regolarizzazione per mini-batch e traiettoria dell'ottimizzatore interagiscono: il confronto non ha budget di aggiornamenti identico e non identifica esclusivamente il contributo informativo della simmetria. Epoche, tempi e curve di loss sono registrati.

## Griglia ristretta di controllo e curve

Per ciascuna famiglia si seleziona anche il migliore candidato fra le tre configurazioni della vecchia griglia, riusando i fit interni della griglia completa. Il controllo usa lo stesso nuovo preprocessing e lo stesso limite di iterazioni, quindi non coincide in ogni dettaglio con la vecchia pipeline archiviata. Consente un confronto appaiato fra griglia ristretta ed estesa sugli stessi fold esterni.

Per la configurazione scelta dalla griglia estesa si misurano errore di training e valutazione con il 50%, 75% e 100% delle squadre del training esterno. I sottoinsiemi sono annidati e deterministici. Gli iperparametri restano fissi lungo la curva; non si riesegue una selezione a ogni numerosità. Sono curve diagnostiche condizionate alla configurazione scelta. La distanza train/validation aiuta a discutere adattamento e generalizzazione senza diagnosticare underfitting o overfitting dalla sola metrica di test.

Si riportano log loss, Brier, accuracy, scelte per fold, avvisi e tempi. Le deviazioni standard dei nove risultati esterni sono descrittive; i fold sovrapposti non sono nove osservazioni indipendenti. Il nuovo confronto è esplorativo, avviato dopo aver conosciuto il risultato del nucleo. Un miglioramento di sviluppo non equivale a una nuova conferma sul test finale.

## Esecuzione remota e ripresa

Host `Francesco-Desktop`, directory `/home/francesco/ICON-experiments/project`. Otto processi con un solo thread BLAS/OpenMP per processo. Preparazione e fit vengono eseguiti sul remoto. Lo script rifiuta l'esecuzione su un host diverso e impedisce due esecuzioni contemporanee tramite lock.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/run_expanded_search.py prepare
.venv/bin/python scripts/run_expanded_search.py pilot --workers 8
.venv/bin/python scripts/run_expanded_search.py run --workers 8
```

La preparazione richiede `reports/expanded_search_v1/local_source_hashes.json`, che verifica la corrispondenza fra codice/dati locali e remoti. Congela protocollo, versioni, hash e matrici in `reports/expanded_search_v1/`. Il pilot misura soltanto il costo di configurazioni prefissate, senza usarne gli score per modificare la griglia.

`status.json` riporta lo stato corrente. Ogni candidato completato nei tre fold interni ha un checkpoint atomico. `summary.json` contiene i fold esterni già completati e un campo `complete` che distingue risultati parziali e conclusi. Lo stesso comando `run` riprende i checkpoint verificati; un errore di fit interrompe l'esecuzione e viene registrato, senza scartare silenziosamente il candidato.

Script nuovo in `scripts/run_expanded_search.py`, implementazione in `src/icon_pokemon/expanded_search.py`, griglia in `config/expanded_search_v1.json`. Il codice e il modello del nucleo congelato rimangono invariati.

## Risultati sullo sviluppo

Il run remoto è terminato il 5 novembre 2026 alle 00:17 (ora italiana), in 60 minuti e 7 secondi. Sono state completate tutte le 6.012 valutazioni di candidati, ciascuna con tre fold interni, e registrati i 216 risultati esterni attesi. Il riepilogo locale è in [summary.json](expanded_search_v1/summary.json). Sono stati verificati l'hash del protocollo, l'unicità delle combinazioni e i nove fold per ciascuna delle 24 condizioni.

Log loss media dei fold esterni per KB v1. La baseline costante a 0,5 vale 0,69315.

| Modello | Ristretta senza augmentation | Estesa senza augmentation | Estesa con augmentation |
|---|---:|---:|---:|
| Logistica | 0,71000 | 0,69069 | 0,69069 |
| Boosting | 0,68932 | 0,68961 | 0,69106 |
| MLP | 0,69042 | 0,69156 | 0,69216 |

La logistica migliora con la griglia estesa. `C=0.001`, il valore più regolarizzato in griglia, è scelto in tutte le 36 selezioni estese della famiglia. Il boosting ristretto con KB v1 e senza augmentation conserva la media più bassa fra le condizioni confrontate. L'augmentation non produce un guadagno consistente nella griglia estesa. Si tratta di confronti descrittivi sullo sviluppo.

Nelle curve della griglia estesa con KB v1 e senza augmentation, passando dal 50% al 100% delle squadre di training, il boosting passa da 0,63009 a 0,65471 sul training e da 0,70313 a 0,68961 sui fold esterni. La MLP passa da 0,64780 a 0,67349 sul training e da 0,69762 a 0,69156 sui fold esterni. Il divario si riduce al crescere delle squadre di training in entrambe le famiglie. Queste curve variano il numero di squadre, quindi non isolano l'effetto delle repliche per coppia.
