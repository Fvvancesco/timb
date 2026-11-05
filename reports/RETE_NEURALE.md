# Rete neurale sullo sviluppo v2–v4

MLP con standardizzazione stimata nel fit, ReLU, Adam, batch 64, learning rate 0,001. Tre configurazioni prefissate: 32 neuroni con L2 alpha 1 o 10, oppure strati 64/32 con alpha 10. Massimo 200 epoche, tol 1e-4, pazienza 20 sulla loss di training. Nessuno split casuale automatico per early stopping. Selezione nei fold interni; stesse composizioni disgiunte, dati e seed 11/23/47 del confronto precedente.

| Modello | Log loss ↓ | Brier ↓ | Accuracy |
|---|---:|---:|---:|
| 50% fisso | 0,69315 | 0,25000 | 50,41% |
| Boosting precedente | 0,68932 | 0,24808 | 54,09% |
| MLP, KB precedente | 0.69044 | 0.24865 | 52.90% |
| MLP, KB estesa | 0.69181 | 0.24933 | 50.86% |

La MLP si avvicina al boosting ma non lo supera in media. La KB estesa non migliora la MLP in questo confronto. Questo riguarda i conteggi delle capacità, non il ruolo della KB nei vincoli e nelle spiegazioni. Non sono stati allenati embedding.

v1: 24/90 fit segnalano il raggiungimento del limite di iterazioni; 0 sono fit finali dei fold esterni. Configurazioni selezionate: {'small_l2_10': 7, 'two_layers_l2_10': 2}. I risultati non vengono eliminati per mancata convergenza; tutti gli avvisi sono nei report.
v2: 18/90 fit segnalano il raggiungimento del limite di iterazioni; 0 sono fit finali dei fold esterni. Configurazioni selezionate: {'small_l2_10': 7, 'two_layers_l2_10': 2}. I risultati non vengono eliminati per mancata convergenza; tutti gli avvisi sono nei report.

Alla conclusione di questo confronto erano passati 37 test automatici, inclusi ripetibilità del fit a seed fisso, standardizzazione e complementarità delle probabilità scambiando i lati. Successivamente è stato congelato il boosting e concluso il test finale: vedere `VALUTAZIONE_FINALE.md`. Questi risultati MLP restano risultati di sviluppo e non modificano il modello congelato.

Riproduzione: `scripts/run_supervised.py --mode nested --experiment neural --kb-version v1` (poi v2), passando a `--labels` i tre `development_labels.jsonl` dei lotti v2, v3, v4 e output distinti.
