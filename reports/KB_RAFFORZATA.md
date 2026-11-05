# KB v2 — effetti strutturati e requisiti di costruzione

Verifica del 4 novembre 2026. Riproduzione dalla radice del progetto:

```bash
.venv/bin/python scripts/run_kb_enrichment.py
.venv/bin/python -m pytest -q
```

## Cosa è cambiato

L'esportatore `tools/showdown/export-move-semantics.cjs` legge la versione locale fissata di Showdown. Produce un file separato con gli effetti strutturati delle mosse. Il catalogo delle battaglie, i set e le etichette non vengono riscritti. La nuova estensione è attivata passando `move_semantics` a `build_graph`; il comando sopra esercita l'intera catena sul solo catalogo train.

Ogni effetto diventa una risorsa RDF collegata alla mossa, con tipo, destinatario, fase primaria/secondaria, campo sorgente e, dove presenti, statistica, variazione di stadi, status e probabilità dichiarata. La probabilità di un effetto secondario non è la probabilità di successo complessivo della mossa. Il grafo registra inoltre i nomi dei callback non interpretati.

Gli assiomi in `ontology/move_semantics.ttl` classificano gli effetti e le mosse, quindi propagano le capacità a set e squadre. Otto capacità si aggiungono alle quattro originarie:

- Potenziamento fisico, speciale, della Velocità e difensivo mediante effetti primari su sé stessi.
- Possibile applicazione di uno status principale al bersaglio, anche come effetto secondario.
- Cambio del possessore della mossa e cambio forzato del bersaglio.
- Disponibilità di Reflect, Light Screen o Aurora Veil.

Sono inferite anche combinazioni: set con potenziamento sia fisico sia speciale, squadra con potenziamento fisico e schermi, squadra con potenziamento speciale e schermi. Le capacità di una squadra possono appartenere a membri diversi. Queste combinazioni descrivono risorse disponibili e non dimostrano una sinergia competitiva.

## Esempi e limiti verificati

Swords Dance supporta il potenziamento fisico. Calm Mind supporta quello speciale e difensivo. Shell Smash supporta fisico, speciale e Velocità, ma non viene classificata come potenziamento difensivo perché riduce le difese. Growl non è un potenziamento del possessore. L'aumento di Velocità secondario di Flame Charge rimane rappresentato senza diventare una capacità di setup primario.

Thunder e Nuzzle conservano rispettivamente la probabilità dichiarata del 30% e del 100% per lo status secondario. Immunità, precisione e condizioni della battaglia possono comunque impedirlo. Aurora Veil è una mossa con effetto schermo e callback `onTry` non interpretato: la KB non certifica che siano soddisfatte le condizioni meteorologiche. I costi e i prerequisiti di Fillet Away non sono ricostruiti dai suoi incrementi di statistiche; i callback sono visibili nelle spiegazioni.

Belly Drum è un esempio di limite: il potenziamento è implementato in un callback e non è dedotto dal nuovo importatore. Assenza di inferenza significa capacità non certificata da questa rappresentazione, non impossibilità dell'effetto. Il report elenca i callback, compresi i casi senza effetti strutturati esportati. Non si attribuisce completezza alla KB; strumenti, abilità e stato dinamico non sono interpretati.

## Collegamento al COP e spiegazioni

Il solver accetta le nuove capacità come requisiti. La richiesta «mantieni questi membri e assicurati di avere setup fisico e schermi» viene tradotta in vincoli sui set certificati dal reasoner. La riparazione conserva la priorità di minimizzare le sostituzioni. Gli obiettivi simbolici precedenti non sono stati ridefiniti implicitamente aggiungendo otto categorie.

La spiegazione risale da requisito a set, mossa ed effetto, includendo bersaglio, fase, campo sorgente e callback non interpretati. È una motivazione della soddisfazione del vincolo, non della probabilità di vittoria.

Il COP usa una lettura operativa chiusa del catalogo: un candidato senza evidenza non può soddisfare il requisito. Questo non trasforma la mancata inferenza OWL in una negazione logica. Un risultato infeasible significa che la richiesta non è certificabile con questo catalogo e queste conoscenze.

## Risultati del collaudo

`reports/kb_enrichment.json` registra hash, conteggi e testimoni. Sono stati utilizzati soltanto i 100 team train: 594 set distinti e 246 mosse. Il grafo ha 10.404 triple asserite e 21.722 triple dopo la chiusura, eseguita in circa 14,8 secondi in questa singola prova locale. Il conteggio della chiusura comprende assiomi di servizio OWL/RDFS e non misura da solo la ricchezza della conoscenza.

Il demo ha riparato una squadra priva di schermi con una sostituzione, provata minima sotto i requisiti e i membri bloccati. La squadra risultante è stata validata da Showdown. È un test di integrazione, non una valutazione competitiva.

La suite completa conta 35 test passati. I nuovi casi controllano direzione e fase degli effetti, conservazione della probabilità, visibilità dei callback, composizione delle inferenze, rimozione di premesse, uso delle deduzioni nella riparazione e rifiuto di metadati incompleti o di una versione diversa del simulatore.

## Confine sperimentale

La rappresentazione supervisionata originaria a 236 caratteristiche rimane invariata. Non sono stati riaddestrati modelli né utilizzati esiti di test in questo intervento. La KB estesa è una variante esplicita da confrontare successivamente, con schema e configurazione registrati. Gli incontri già raccolti rimangono riutilizzabili: le etichette dipendono dai set simulati e dal pilota, che non sono cambiati.
