# Previsione del sesto membro da replay umani

L'esperimento usa i due shard Gen 9 OU di PokéChamp e Metamon già usati per il Team Preview. Riutilizza il filtro dei replay decisivi con rating minimo 1600 per entrambi i giocatori. Il vincitore serve al filtro storico, mentre le caratteristiche e il target provengono esclusivamente dalla composizione. Le forme vengono raggruppate per specie base. Il campione dà lo stesso peso a ogni composizione unica, quindi la baseline di frequenza misura la popolarità tra composizioni distinte nel training.

Lo split con seed 307 assegna 4.643 composizioni al training, 995 alla validation e 996 al test. Ogni composizione produce sei esempi, ciascuno con un membro mascherato. Il vocabolario comune contiene 331 specie di training. I dieci target fuori vocabolario contano come errori top-k. La log loss usa i 5.966 target in vocabolario, gli stessi per tutti i metodi. I membri visibili già noti sono esclusi dalle proposte.

| Metodo | Top-1 | Top-5 | Log loss sui target in vocabolario |
|---|---:|---:|---:|
| Frequenza | 7,71% | 22,76% | 4,4143 |
| Naive Bayes | 14,96% | 37,08% | 4,4837 |
| Pokémon2Vec | 20,48% | 45,13% | 3,5923 |

Naive Bayes migliora il top-5 rispetto alla frequenza di 14,32 punti percentuali, IC 95% [12,60; 16,05]. La differenza di log loss è +0,0695, IC [−0,0262; +0,1673]. Pokémon2Vec migliora il top-5 rispetto a Bayes di 8,05 punti, IC [6,78; 9,30]. La differenza di log loss rispetto a Bayes è −0,8914, IC [−0,9733; −0,8130]. Gli intervalli percentile appaiati usano 10.000 ricampionamenti per composizione, con seed 11 e le sei maschere mantenute insieme.

I parametri sono stati scelti per top-5 in validation, con log loss come spareggio. Bayes usa smoothing 0,1. Pokémon2Vec è un predittore del membro nascosto con embedding da 32 dimensioni, media dei cinque vettori, ReLU e classificazione sulle specie. Media le probabilità di tre fit a 35 epoche con seed 11, 23 e 47. I risultati riguardano composizioni riservate tratte dagli stessi shard. La verifica su un periodo o una fonte indipendente rimane un esperimento separato.

- Split e hash delle fonti in `data/processed/team_completion_v1/splits.json`.
- Griglie di validation, configurazioni congelate e vocabolario in `selection.json`.
- Metriche, intervalli e predizioni in `report.json` e `test_predictions.jsonl`.
- Codice di estrazione e valutazione in `scripts/run_team_completion_extension.py`. Per preparare lo split eseguire `prepare POKECHAMP.parquet METAMON.parquet` con PyArrow disponibile. Con lo split salvato, il comando `evaluate` usa le dipendenze bloccate del progetto.

## Verifica esplorativa con CP-SAT

Abbiamo selezionato Pokémon2Vec in base alla validation. Su 100 squadre di test distinte, con seed 719, abbiamo estratto una maschera per avversario e cinque membri da bloccare in una squadra propria del catalogo train. Entrambe le varianti ricevono lo stesso catalogo e i cinque membri avversari visibili. La prima usa il completamento più probabile. La seconda normalizza le prime cinque probabilità e le approssima con 20 unità intere, assegnando almeno un'unità a ogni ipotesi. Abbiamo riusato CP-SAT con colonne di copertura replicate secondo il peso e validato ogni squadra con Showdown.

| Strategia | Squadre legali e ottime | Specie nascosta coperta | Risposte offensive medie | Specie visibili coperte |
|---|---:|---:|---:|---:|
| Ipotesi più probabile | 100/100 | 97/100 | 2,73 | 5,00/5 |
| Cinque ipotesi pesate | 100/100 | 99/100 | 2,81 | 5,00/5 |

La differenza appaiata di copertura del membro nascosto è +2 punti percentuali, IC bootstrap 95% [0; +5]. L'intervallo include zero e la baseline è già vicina al massimo. La metrica indica almeno una mossa superefficace contro il tipo canonico della specie base. I replay mascherati forniscono la specie, mentre il suo set effettivo resta sconosciuto. I risultati descrivono la copertura statica.

Le soluzioni, le misure per richiesta e gli hash sono in `solver_scenarios.json`. Per ripetere il confronto dopo aver generato le predizioni, eseguire `.venv/bin/python scripts/run_completion_solver_extension.py`.

## Configurazioni di Pokémon2Vec sulla validation

La rappresentazione di partenza è un bag delle cinque specie visibili. Il primo strato equivale alla media dei cinque vettori appresi, seguita da ReLU e da uno strato di classificazione. Tutte le configurazioni usano la media delle probabilità di tre seed. La scelta usa il top-5, con log loss come spareggio.

| Dimensioni | Epoche | Top-1 | Top-5 | Log loss |
|---:|---:|---:|---:|---:|
| 16 | 5 | 12.78% | 33.15% | 4.0554 |
| 16 | 10 | 16.75% | 40.50% | 3.8004 |
| 16 | 20 | 18.99% | 43.47% | 3.6734 |
| 16 | 35 | 19.35% | 44.25% | 3.6530 |
| 32 | 5 | 14.92% | 37.99% | 3.9184 |
| 32 | 10 | 18.54% | 43.50% | 3.6882 |
| 32 | 20 | 20.03% | 45.18% | 3.6109 |
| 32 | 35 | 20.70% | 45.43% | 3.5960 |
