# Reinforcement Learning per la gestione del traffico in un sistema di Edge Computing 

## Descrizione
Questo progetto propone l'implementazione di un algoritmo di apprendimento per rinforzo, Deep Deterministic Policy Gradient (DDPG), per ottimizzare la gestione di carichi di lavoro in un sistema di elaborazione distribuito. L'obiettivo è trovare la policy ottimale per l'elaborazione locale, l'inoltro delle richieste a nodi edge e il rifiuto di richieste in base alle condizioni del sistema. 
L'implementazione attuale presenta ancora ipotesi semplificatrici rispetto lo scenario reale.

## Ambiente
L'ambiente simula un sistema di elaborazione distribuito con una capacità di elaborazione locale massima e una coda per gestire le richieste in arrivo. Ad ogni nuovo episodio, l'ambiente viene resettato con le seguenti condizioni:

Carico di lavoro: 0
Capacità di CPU: massima (50 unità)
Capacità della coda: massima (100 richieste)
Le richieste vengono generate secondo una funzione sinusoidale con un minimo di 50, un massimo di 150 e un periodo di 99. Si assume che tutte le richieste richiedano la stessa quantità di CPU.

Uno "step" termina quando la coda si riempie, indicando una congestione del sistema. Al termine di ogni step, vengono aggiornate le informazioni sullo stato del sistema, incluse la capacità della CPU e la capacità della coda.

L'obiettivo è favorire l'elaborazione locale a meno che la coda non sia quasi piena. In tal caso, per evitare la congestione, l'agente deve inoltrare le richieste.

## DDPG
L'algoritmo DDPG è implementato in TensorFlow. I parametri dell'algoritmo non sono stati ottimizzati attraverso una specifica tecnica, ma attraverso vari tentativi di addestramento.

Viene definito un buffer di riproduzione (Replay Buffer) da cui vengono campionate le esperienze. Per evitare episodi infiniti in caso di politiche ottimali, viene fissato un massimo di 50 steps per episodio.

Per favorire l'esplorazione in un contesto deterministico, viene introdotto rumore di Ornstein-Uhlenbeck (OU) che modifica l'output della rete attore. Il valore di sigma per il rumore di OU inizia da un livello relativamente alto (0.30) e decade nel corso degli episodi.
