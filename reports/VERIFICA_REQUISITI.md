# Verifica dell'impianto rispetto al materiale del corso

## Aggiornamento alla chiusura del progetto

La verifica sotto è il documento di pianificazione del 3 novembre e mantiene le indicazioni originarie. Alla revisione finale del 4 novembre, inferenze relazionali, CV annidata e benchmark sono implementati e documentati nella [relazione](../RELAZIONE_PROGETTO.md), sezioni 3–8. L'estensione della sezione 9 comprende Naive Bayes, K-means, Pokémon2Vec e RDF2Vec. Gli embedding non sono più rinviati. Il test finale del nucleo è concluso e non più riservato per nuove scelte.

La valutazione del nucleo usa più partizioni annidate; le estensioni usano uno split originale e due successivi holdout di composizioni per i confronti specifici. I tre seed della rete non sostituiscono tre split, e gli shard esterni non sono una CV annidata delle estensioni. Questo confine va mantenuto nella consegna. Le [istruzioni di riproduzione](../RIPRODUCIBILITA.md) descrivono le verifiche tecniche disponibili.

## Verifica originaria

Verifica del 3 novembre 2026. Fonti locali: programma Fanizzi 2025/26, presentazione dell'insegnamento, schermate di linee guida e FAQ, relazioni Semafori Intelligenti, Chicago Crimes e Gestione delle playlist di Spotify. Questo giudizio riguarda l'allineamento tecnico ai documenti forniti, non sostituisce il vaglio di ammissibilità della commissione. Non abbiamo verificato aggiornamenti successivi dei siti del corso.

## Esito

L'impianto è coerente e sufficientemente articolato per procedere, incorporando due correzioni obbligatorie nel piano: ragionamento relazionale della KB più sostanziale delle sole quattro etichette di set e valutazione su più partizioni, con selezione degli iperparametri separata dalla stima delle prestazioni. Il prototipo precedente, da solo, non era ancora una consegna adeguata.

## Criteri e risposte progettuali

| Criterio documentato | Risposta del progetto | Evidenza da consegnare |
|---|---|---|
| Complessità e completezza, diversità degli argomenti | RDF/OWL e inferenza, COP con riparazione lessicografica, classificazione probabilistica, valutazione dell'agente | Formalizzazione, catena dei dati e confronto delle componenti |
| KB non ridotta a DB o pattern matching sui fatti | Proprietà concatenate squadra-membro-mossa, restrizioni esistenziali, intersezioni, deduzioni con testimoni | Domande di competenza e derivazioni assenti dal grafo asserito, controesempi, tempi e dimensione della chiusura |
| Qualità della valutazione | Baseline, CV annidata per composizioni di squadra, test finale riservato, ripetizioni del solver e simulazioni appaiate | Tabelle con media e deviazione standard per split/run e intervalli per i confronti finali, conteggi effettivi |
| Originalità, riuso ammesso | Integrazione di Showdown, OWL RL, CP-SAT e modelli per completamento/riparazione spiegabile | Contributo proprio distinto da solver, simulatore e pilota riusati; confronto con lavori precedenti |
| Generalità | Catalogo e richieste parametrizzati, membri bloccati, soglie e requisiti variabili | Più richieste di costruzione/riparazione e più avversari; nessuna promessa di trasferimento già provato fra generazioni |
| Decisioni motivate e limiti | Selezione supervisionata valutata esternamente, copertura statica distinta da vittoria | Risultati anche negativi, impatto del pilota, esclusioni e casi non conclusi |
| Impegno individuale | Nucleo finito, riuso degli strumenti, niente clustering ornamentale | Rimandare embedding, nuovi piloti e regole complete sugli effetti |

Il programma include esplicitamente vincoli/ottimizzazione (cap. 4), rappresentazione e ragionamento relazionale (cap. 15), knowledge graph/ontologie (cap. 16), classificazione, regolarizzazione, cross-validation e boosting (cap. 7). Probabilità predette e simulazioni non equivalgono ad aver implementato reti bayesiane, HMM o inferenza probabilistica del cap. 9. Nessuna fonte locale esaminata impone di implementare tutti i capitoli o clustering.

Le linee guida chiedono di soffermarsi sulla rappresentazione, sull'uso e sulla complessità di assiomi/regole. Va spiegato il confine fra mondo aperto di OWL e catalogo chiuso del COP: una capacità non inferita non è una negazione logica; il solver può selezionare solo capacità certificate nel catalogo dichiarato completo per il compito. I moltiplicatori sono dati del simulatore; prodotto su doppi tipi e soglie sono calcoli Python, non deduzioni attribuite a OWL. Descrivere anche variabili, vincoli, limiti del punteggio e costi di inferenza/ricerca.

Le linee guida criticano risultati da un unico run/test set e indicano medie e deviazioni standard da CV annidata su diversi test set. I seed di addestramento o le repliche di battaglia non sostituiscono nuove partizioni dei dati. Il documento precedente sottovalutava questo punto: la sola divisione train/validation/test è insufficiente come intero impianto sperimentale.

## Confronto con le relazioni dei compagni

- **Semafori Intelligenti** (2022/23, quattro autori): integra conoscenza estratta da OpenStreetMap, Prolog, regressione, HMM, CSP e ricerca. Nella sezione CSP (pagine stampate 25–28) confronta qualità e tempi e usa generate-and-test per ottenere un riferimento ottimo. Da riprendere l'integrazione e l'oracolo su piccoli casi. L'ampiezza di un lavoro di quattro persone non è il minimo necessario per un progetto individuale.
- **Chicago Crimes** (2022/23, due autori): inferisce feature da una KB Prolog e confronta più classificatori, riportando risultati mediati su dieci modelli nella sezione sulle metodologie. Il clustering degli embedding è un esperimento separato e la conclusione non trova differenze significative fra i cluster. Il parallelo utile è KB → informazione derivata → apprendimento, con risultati discutibili criticamente. Non occorre replicare il numero di algoritmi.
- **Gestione delle playlist di Spotify** (2023/24, un autore, file Documentazione.pdf): clustering per creare playlist, classificazione dell'indice del cluster, reti bayesiane e interrogazioni Prolog. Mostra un progetto individuale con più temi. Il nostro target ha invece una verifica esterna mediante esiti del simulatore; questo costituisce una differenza di disegno, non una prova che il nostro progetto avrà prestazioni o voto migliori.

Non conosciamo i voti né i giudizi del docente su queste relazioni. Sono esempi di struttura e scelte, non rubriche normative.

## Correzioni operative

1. Implementare subito inferenze su squadre, combinazioni di capacità e tracciabilità membro/mossa; testare rimozione di premesse, assenza di inferenza e casi limite. Collegare queste risposte al COP e alla spiegazione.
2. Prima del manifest esteso, preregistrare CV annidata a tre fold esterni e tre interni sul pool di sviluppo. Raggruppare per composizione di squadra: una battaglia è utilizzabile nel train solo se entrambe le squadre appartengono al train del fold, e nel test solo se entrambe appartengono al test. Escludere dal fold gli incontri che attraversano le partizioni. Campionare le coppie tenendo conto di questa struttura e riportare le perdite di dati.
3. Verificare numerosità e presenza delle due classi nei fold prima del fit; se insufficienti, raccogliere sviluppo aggiuntivo o ridimensionare esplicitamente il protocollo prima di vedere il test finale. Non presentare stime instabili come risultati conclusivi. La CV riusa le simulazioni raccolte, non richiede di ripetere ogni battaglia nove volte.
4. I 20 team test v1 restano sigillati per la valutazione finale; train e validation v1 formano il pool di sviluppo per la CV annidata. Iperparametri nelle partizioni interne, prestazioni nelle esterne; refit finale sullo sviluppo con selezione interna. Riportare media e deviazione standard dei fold, senza considerarli campioni indipendenti per test statistici ingenui.
5. Quantificare scala della KB (set/team/triple/assiomi, inferenze e tempo) e scala del COP (candidati, variabili, vincoli, qualità e bound a budget crescente).
6. Documentare l'agente: osserva richiesta e squadra avversaria completa, produce selezione/riparazione; il pilota di battaglia è un componente esterno di valutazione. Distinguere modello statico del costruttore e ambiente stocastico del simulatore.

Il budget da 1.000 partite è una stima iniziale da ricalcolare sul manifest compatibile con i fold. Non avviare un lotto esteso con il vecchio schema casuale senza questa verifica.
