# Confronto supervisionato sullo sviluppo v2–v4

5.463 esiti decisivi, stesse partizioni per composizione, CV annidata 3×3 con seed 11/23/47. Media e deviazione standard sui nove fold esterni. La deviazione standard descrive la dispersione: non è un intervallo di confidenza, perché i fold condividono osservazioni.

| Metodo | Rappresentazione | Log loss ↓ (media ± SD) | Brier ↓ (media ± SD) | Accuracy (media ± SD) |
|---|---|---:|---:|---:|
| train_rate | v1 | 0.69336 ± 0.00022 | 0.25010 ± 0.00011 | 49.70% ± 2.00 pp |
| neutral | v1 | 0.69315 ± 0.00000 | 0.25000 ± 0.00000 | 50.41% ± 1.98 pp |
| logistic | v1 | 0.71001 ± 0.02641 | 0.25689 ± 0.01173 | 53.13% ± 3.31 pp |
| boosting | v1 | 0.68932 ± 0.00582 | 0.24808 ± 0.00287 | 54.09% ± 2.79 pp |
| logistic | v2 | 0.71260 ± 0.02877 | 0.25817 ± 0.01263 | 52.57% ± 2.51 pp |
| boosting | v2 | 0.69221 ± 0.00675 | 0.24953 ± 0.00334 | 52.25% ± 2.93 pp |
| mlp | v1 | 0.69044 ± 0.00570 | 0.24865 ± 0.00283 | 52.90% ± 2.67 pp |
| mlp | v2 | 0.69181 ± 0.00694 | 0.24933 ± 0.00344 | 50.86% ± 3.69 pp |
| decision_tree | v1 | 0.69590 ± 0.00943 | 0.25127 ± 0.00456 | 52.60% ± 2.00 pp |
| random_forest | v1 | 0.68706 ± 0.00648 | 0.24697 ± 0.00321 | 55.06% ± 3.05 pp |
| decision_tree | v2 | 0.69763 ± 0.00809 | 0.25214 ± 0.00390 | 51.36% ± 0.89 pp |
| random_forest | v2 | 0.68777 ± 0.00608 | 0.24732 ± 0.00300 | 53.72% ± 3.61 pp |

La random forest v1 ottiene la log loss media più bassa di questo confronto aggiuntivo. Non sostituisce il boosting già congelato e valutato nel test finale. Per una conferma indipendente di una futura versione con random forest servirebbe un nuovo protocollo e un nuovo test riservato.

v1: 236 caratteristiche, incluse quattro capacità originarie della KB. v2: 252 caratteristiche, con dodici capacità. Il confronto riguarda i conteggi usati dal predittore, non elimina la KB dai vincoli e dalle spiegazioni. L’estensione non migliora in media nessuna delle famiglie provate.

Albero: profondità 2/4/8, almeno 20 osservazioni per foglia. Random forest: 200 alberi, profondità 4/8/illimitata, almeno 10 osservazioni per foglia, max_features=sqrt, bootstrap, quattro processi/thread di lavoro. Scelta della profondità solo nei fold interni. Probabilità simmetrizzate come negli altri modelli. Nessuna nuova griglia è stata aggiunta dopo questi risultati.

Per riprodurre il riepilogo: `.venv/bin/python scripts/summarize_development.py`. Per rifare gli esperimenti aggiuntivi sul remoto: `.venv/bin/python scripts/probe_tree_models.py --kb-version v1`, poi `v2`. Non eseguire training mentre il benchmark dei tempi usa lo stesso host.

L’accuracy della costante p=0,5 usa la soglia >=0,5 e quindi predice sempre la stessa classe: non rappresenta un classificatore che estrae casualmente le etichette.
