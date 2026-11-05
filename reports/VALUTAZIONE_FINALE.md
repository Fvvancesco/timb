# Valutazione finale del sistema congelato

Il test del nucleo è completato: **312 battaglie fisiche**, 307 concluse, tre timeout e due interruzioni al limite dei turni. Nessuna richiesta di costruzione è fallita. I risultati non dimostrano un vantaggio competitivo del selettore supervisionato rispetto alla scelta casuale dal medesimo pool. Questa conclusione non annulla la verifica di legalità, vincoli, inferenze e riparazione minima: sono proprietà diverse, valutate separatamente.

## Congelamento e unità sperimentali

Il modello è stato scelto esclusivamente sui 5.463 esiti decisivi di sviluppo: gradient boosting, profondità massima 1, rappresentazione v1 a 236 caratteristiche. Sono stati salvati modello, schema, versioni e hash. Prima delle battaglie sono state congelate 80 predizioni, 20 richieste e tutte le squadre selezionate. `data/processed/final_v1/evaluation_started.json` registra l’avvio della valutazione. Il test è ora **già utilizzato**: non può essere riusato per scegliere un modello migliore dichiarando una nuova conferma indipendente.

Le 80 battaglie predittive riguardano 40 coppie non ordinate di squadre test. Le altre 232 valutano tre selettori su 20 richieste, dieci completamenti e dieci riparazioni, ciascuna con un avversario riservato. Sono previste due radici (401, 409) e due orientazioni per metodo/richiesta. Scelte identiche condividono le stesse partite: per questo 3×20×4 osservazioni logiche corrispondono a 232 battaglie fisiche.

Tutti i selettori ricevono lo stesso pool di 20 squadre prodotto da CP-SAT, con gli stessi membri fissi, capacità richieste e catalogo. Nelle dieci riparazioni è provata la sufficienza e minimalità di una sostituzione. Il selettore casuale di questo esperimento sceglie **nel pool già costruito**: è diverso dal costruttore casuale legale del benchmark di ricerca.

## Predizione prima della battaglia

| Metodo | Log loss ↓ | Brier ↓ | Accuracy |
|---|---:|---:|---:|
| Costante p=0,5 | 0,69315 | 0,25000 | 40,00% |
| Boosting congelato | 0,69737 | 0,25210 | 52,50% |

Il Brier è la media di `(p−y)²`, con y uguale a 0 oppure 1: penalizza probabilità lontane dall’esito osservato. La log loss penalizza particolarmente le previsioni sicure ma sbagliate. Entrambe sono migliori quando diminuiscono. L’accuracy del 50% fisso deriva dalla regola `p >= 0,5`: qui predice sempre A e A vince 32 volte su 80. Non è l’accuracy attesa di una moneta casuale; scegliere la classe maggioritaria dopo aver visto il test sarebbe una baseline costruita a posteriori.

Sul test originale l’AUROC è 0,486 (IC 95% [0,354; 0,621]). Alla soglia 0,5, la matrice di confusione con righe reali e colonne predette nell’ordine sconfitta/vittoria della squadra A è `[[25, 23], [15, 17]]`.

Bootstrap percentile con 10.000 repliche, seed 11, raggruppando le due orientazioni della stessa coppia:

- Differenza di log loss, modello meno costante: **+0,00423**, IC 95% **[−0,01300; +0,02241]**.
- Differenza di Brier: **+0,00210**, IC 95% **[−0,00646; +0,01114]**.
- Accuracy del modello: 52,50%, IC 95% [41,25%; 63,75%].

Il piccolo vantaggio medio di sviluppo non è confermato sul test predittivo. Le coppie condividono alcune squadre: il raggruppamento conserva le repliche ma non elimina tutta la dipendenza fra coppie. Gli intervalli non vanno interpretati come una garanzia di copertura esatta in questo disegno.

## Replica su nuove squadre

Abbiamo deciso di replicare il test predittivo dopo avere osservato il primo risultato. Prima dei nuovi esiti abbiamo fissato modello, campione, seed e metriche. La replica usa 100 coppie disgiunte formate da 200 composizioni Gen 9 OU legali di Metamon, assenti dal catalogo originale e dai precedenti campioni di avversari per la copertura. Per ogni coppia si giocano entrambe le orientazioni con Foul Play in self-play. Dei 200 job pianificati, 200 producono un esito decisivo verificato. Conteggi per stato nel [riepilogo della replica](../data/processed/prediction_replication_v1/summary.json).

| Metrica | Boosting congelato | Costante p = 0,5 |
|---|---:|---:|
| Log loss | 0,69205 | 0,69315 |
| Brier | 0,24938 | 0,25000 |
| Accuracy a 0,5 | 52,50% | 48,50% |
| AUROC | 0,554 | 0,500 |

La differenza primaria di log loss è −0,00110, IC 95% [−0,01611; 0,01516]. Per il Brier la differenza è −0,00062, IC 95% [−0,00803; 0,00737]. L’IC dell’AUROC è [0,462; 0,646]. Il bootstrap usa 10.000 ricampionamenti per coppia con seed 11. Alla soglia 0,5 la matrice, con righe reali e colonne predette nell’ordine sconfitta/vittoria della squadra A, è `[[54, 49], [46, 51]]`.

La replica è un test separato del predittore. Non rivaluta i selettori del costruttore. Le cinque assegnazioni hanno usato 36 worker CPU in totale. La ricerca del pilota è a tempo e resta sensibile all’hardware. Su quattro host l’accesso al server locale usa il login guest diretto. La configurazione di ricerca è 25 ms e un thread come nel test originale.

## Ultima replica su 600 squadre

Il terzo lotto usa lo stesso modello congelato. Le 600 composizioni legali sono nuove rispetto ai due test predittivi e al catalogo originale. Formano 300 coppie disgiunte, ognuna giocata nei due orientamenti. Campione, seed e analisi sono stati fissati prima della simulazione. I tre campioni sono riportati separatamente. Delle 600 partite, 597 concluse, 3 al limite dei turni. Gli esiti decisivi verificati sono 597 e provengono da 299 coppie.

| Metrica | Boosting congelato | Costante p = 0,5 |
|---|---:|---:|
| Log loss | 0,68850 | 0,69315 |
| Brier | 0,24771 | 0,25000 |
| Accuracy alla soglia 0,5 | 52,93% | 49,75% |
| AUROC | 0,545 | 0,500 |

La differenza di log loss, metrica primaria, è −0,00465 (IC 95% [−0,01189; 0,00256]). L’IC 95% dell’AUROC è [0,496; 0,595]. I due intervalli usano 10.000 ricampionamenti per coppia con seed 11. La matrice di confusione alla soglia 0,5, con righe reali e colonne predette nell’ordine sconfitta/vittoria di A, è `[[159, 141], [140, 157]]`.

L’intervallo della differenza di log loss comprende zero. L’intervallo dell’AUROC comprende 0,5. L’esperimento valuta le probabilità del predittore, mentre il confronto dei selettori resta quello sulle 20 richieste. Dal congelamento al riepilogo sono trascorsi 13 minuti e 18 secondi, con cinque PC e 36 worker. [Manifest](../data/processed/prediction_replication_v2/manifest.json) e [riepilogo](../data/processed/prediction_replication_v2/summary.json).

## Squadre selezionate in battaglia

| Selettore | Vittorie / esiti decisivi | Media delle win rate per richiesta | IC bootstrap 95% | Esiti censurati |
|---|---:|---:|---:|---:|
| Casuale nel pool | 36/80 | 45,00% | [33,75%; 56,25%] | 0 |
| Simbolico | 30/77 | 38,75% | [25,00%; 53,75%] | 3 |
| Supervisionato | 35/78 | 45,42% | [33,75%; 57,50%] | 2 |

La statistica principale dà peso uguale a ogni richiesta. Con esiti mancanti può differire dalla frazione complessiva di vittorie: 35/78 è 44,87%, mentre la media per richiesta è 45,42%. Il bootstrap ricampiona le 20 richieste conservando al loro interno radici e orientazioni.

Nei confronti appaiati si usano solo gli slot conclusi per entrambi i metodi. Per questo la differenza appaiata non coincide necessariamente con la sottrazione delle medie della tabella.

| Confronto appaiato | Differenza media | IC bootstrap 95% | Slot comuni decisivi |
|---|---:|---:|---:|
| Supervisionato − simbolico | +5,00 pp | [−10,83; +20,00] pp | 76 |
| Supervisionato − casuale | +0,83 pp | [−15,42; +18,75] pp | 78 |
| Simbolico − casuale | −5,00 pp | [−21,25; +11,25] pp | 77 |

Tutti gli intervalli includono zero e sono ampi: non abbiamo dimostrato né superiorità né equivalenza fra selettori. L’unità informativa è soprattutto la richiesta, non ciascuna delle molte righe del dataset.

## Test statistici aggiuntivi e censura

Analisi **esplorativa**, specificata dopo l’ispezione dei risultati finali, senza modificare modello o squadre: test bilaterale di inversione del segno sulle 20 differenze per richiesta, 99.999 permutazioni Monte Carlo, seed 11. I p-value sono 0,58314, 1,00000 e 0,65654 nell’ordine della tabella. Dopo correzione conservativa di Bonferroni per i tre confronti sono tutti 1. Non sono probabilità che un metodo sia migliore o che l’ipotesi nulla sia vera.

Il test assume blocchi indipendenti e scambiabilità dei metodi all’interno del blocco sotto l’ipotesi nulla; quest’ultima è più forte della sola uguaglianza delle medie. La condivisione di set e squadre può rendere approssimativa l’indipendenza. L’implementazione usa [`scipy.stats.permutation_test`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), mantenendo appaiate le osservazioni. Non applichiamo un t-test ordinario ai nove fold sovrapposti della CV: la dipendenza può sottostimare l’incertezza, come discusso nell’[esempio ufficiale scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html).

Assegnando ai soli esiti mancanti alternativamente tutte sconfitte o tutte vittorie, la media sul campione completo sarebbe: casuale 45,00%; simbolico da 37,50% a 41,25%; supervisionato da 43,75% a 46,25%. Questi sono **limiti di sensibilità ai dati mancanti, non intervalli di confidenza**. Non risolvono l’incertezza sulle nuove richieste. I timeout restano censurati nell’analisi principale.

## Limiti e interpretazione

Le probabilità descrivono il pilota Foul Play, il suo budget di ricerca di 25 ms, Showdown 0.11.11 e il formato gen9ou fissato. La ricerca nativa a tempo può variare anche con seed del motore e di Python controllati. Non si misura la forza universale di una squadra, né il rendimento con giocatori umani o in VGC.

Il predittore usa aggregati: perde parte dell’associazione fra membri, mosse e statistiche e non codifica direttamente la semantica di strumenti, abilità e Tera. Nel simulatore questi effetti sono invece attivi. È una limitazione plausibile da studiare sullo sviluppo, non una spiegazione causale già provata del risultato negativo. Il pool è già filtrato dal punteggio simbolico: questo restringe il confronto fra selettori e può ridurre il margine di miglioramento.

La random forest successivamente confrontata sul solo sviluppo è promettente, ma non sostituisce retroattivamente il modello di questo esperimento. Un eventuale nuovo test deve essere progettato e riservato prima di osservarne gli esiti.

## Artefatti e riproduzione

Risultati numerici: `reports/final_evaluation.json`, `reports/final_statistical_tests.json`. Input congelati: `data/processed/final_v1/manifest.json`, `predictions.json`, `requests.json`, `catalog.json`, `labels.jsonl`. Modello: `models/selector_development_v1.joblib`. Selezione sullo sviluppo: `reports/development_model_selection.json`.

Ricalcolare le metriche senza generare partite o allenare modelli:

```bash
.venv/bin/python scripts/evaluate_final.py
.venv/bin/python scripts/analyze_final_statistics.py
```

Il secondo comando verifica gli hash degli input dell’analisi finale. Il primo controlla completezza, metadati delle partite, replay verificati e integrità delle predizioni congelate.
