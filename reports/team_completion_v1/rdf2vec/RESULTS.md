# RDF2Vec per il completamento della squadra

La validation ha scelto gli embedding a 32 dimensioni, con top-5 del 45,88% contro il 45,39% per 16 dimensioni. Il grafo proiettato contiene 6.742 triple e 1.286 nodi. Abbiamo generato 64 passeggiate di quattro archi per ciascuna delle 128 specie base, per 8.192 sequenze.

Il test comprende 2.883 composizioni e 17.298 maschere. Sono state escluse 2.387 composizioni già usate. I 40 target fuori vocabolario contano come errori top-k. La log loss usa i 17.258 casi comuni in vocabolario.

| Variante | Top-1 | Top-5 | Log loss |
|---|---:|---:|---:|
| Pokémon2Vec con sole identità | 18,19% | 41,83% | 3,7477 |
| Pokémon2Vec con capacità esplicite | 18,49% | 42,07% | 3,7375 |
| Pokémon2Vec con RDF2Vec | 18,41% | 42,00% | 3,7416 |

Il confronto principale RDF2Vec contro sole identità stima +0,18 punti di top-5, con IC bootstrap appaiato per composizione da −0,19 a +0,55. Contro i profili espliciti la differenza è −0,06 punti, IC da −0,42 a +0,31. Anche la differenza di log loss fra RDF2Vec e sole identità include zero (−0,00617, IC [−0,01295; +0,00061]). I profili espliciti migliorano di circa 0,010 la log loss rispetto alle sole identità, con IC [−0,01684; −0,00362]. Gli intervalli per i confronti secondari sono descrittivi. Sul top-5, i risultati restano compatibili con differenze piccole in entrambe le direzioni. Lo shard riserva composizioni, mentre giocatori e periodo possono sovrapporsi con i dati precedenti.

Le passeggiate contengono sia i nodi sia i predicati. Il grafo è proiettato sul catalogo train e usa relazioni dirette e inverse. Gli embedding sono normalizzati e mediati sui membri visibili. Le tre varianti usano lo stesso addestramento supervisionato e gli stessi split.

Vedere [protocollo](../../../config/rdf2vec_external_v1.json), [selezione](selection.json), [risultati con intervalli](report.json) e [riproducibilità](../../../RIPRODUCIBILITA.md).
