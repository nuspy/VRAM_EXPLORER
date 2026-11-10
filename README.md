# VRAM Explorer

Esplorazione e gestione della memoria video su Windows: **chi** occupa la VRAM,
**quanto**, **cosa** ci tiene dentro, e cosa si può fare al riguardo.

Il servizio è in **sola osservazione**: rileva, registra e segnala, ma non
libera né termina nulla di propria iniziativa. Ogni azione parte da un comando
esplicito.

---

## Il problema che risolve

Su GeForce con driver WDDM — cioè su ogni PC Windows con una scheda NVIDIA
consumer — `nvidia-smi` non sa dire quanta memoria usi ciascun processo:

```
Used GPU Memory : Not available in WDDM driver model
```

Task Manager conosce i totali, ma non cosa ci sia dentro: mostra `python.exe
12 GB` senza dire quale modello, quale quantizzazione, quale riga di codice.

VRAM Explorer legge i contatori di prestazione `GPU Process Memory` (la stessa
fonte del Task Manager) per l'attribuzione per processo, li incrocia con NVML per
lo stato della GPU, e interroga i backend AI per sapere cosa stiano tenendo
caricato.

---

## Cosa mostra

| Livello | Fonte | Informazioni |
|---|---|---|
| GPU | NVML | memoria residente, BAR1, utilizzo SM e memory controller, encoder/decoder, clock, temperatura, potenza, throttling, PCIe |
| Processo | contatori PDH + `Win32_Process` | memoria dedicata, **condivisa**, committed, per adapter e per segmento; attività per motore (3D, Compute, Copy, Video); riga di comando, utente, editore, icona |
| Contenuto | API dei backend, header dei file, riga di comando | modello caricato, quantizzazione, contesto, layer effettivamente in GPU, architettura, teste KV |
| Allocazioni | agent PyTorch (facoltativo) | ogni blocco con dimensione e **stack Python** che l'ha creato, frammentazione, retry, OOM |

### Due numeri diversi, non intercambiabili

La somma della memoria per processo **supera** regolarmente la capacità della
GPU. Non è un errore: Windows conta memoria *committed*, che include pagine che
possono essere sfrattate verso la RAM. NVML riporta invece la memoria
*residente*. L'interfaccia mostra entrambe separatamente e usa il totale
dell'adapter come denominatore.

### Il rilevatore di spillover

Quando `Su RAM di sistema` è maggiore di zero per un processo, quel processo sta
accedendo alla memoria attraverso il bus PCIe invece che alla VRAM. Non produce
errori: solo lentezza, fino a due ordini di grandezza. È il difetto più difficile
da diagnosticare senza uno strumento che lo mostri, e nessun monitor comune lo
espone per processo.

---

## Cosa permette di fare

**Scaricare contenuto dalla VRAM.** Non esiste una API di sistema per liberare
l'allocazione di un altro processo — è il vincolo fondamentale. Si può però
chiedere al processo di farlo, con una scala di garbo decrescente:

1. **API del backend** — LM Studio, Ollama, ComfyUI, Automatic1111. L'unica via
   che libera davvero lasciando il processo vivo.
2. **Cache di PyTorch** — `empty_cache()` tramite l'agent; non tocca i tensori vivi.
3. **Riduzione del working set**, **sospensione**, **terminazione** — su conferma
   esplicita, e mai per i processi di sistema.

**Riservare VRAM.** Un processo dedicato alloca e trattiene la quantità
richiesta. È efficace contro i programmi CUDA, che troveranno meno memoria
libera; **non** è un blocco hardware e non impedisce al compositore di Windows di
ottenere memoria. Il processo di riserva consuma inoltre alcune centinaia di MB
per il proprio contesto CUDA, riportati separatamente.

**Scegliere fra RAM di sistema e fallimento.** Il driver NVIDIA per default
continua allocando nella RAM quando la VRAM finisce. L'alternativa è farlo
fallire subito: per un training è quasi sempre meglio un errore in due secondi
che un degrado di cinquanta volte scoperto la mattina dopo. Il valore aggiunto
rispetto al pannello NVIDIA è il **profilo per applicazione**: fail-fast solo su
`python.exe`, comportamento permissivo per giochi e browser.

**Pianificare.** «Ci sta un 70B Q4 con 32k di contesto?» Calcola pesi, cache KV
(con GQA gestita correttamente: usare `n_heads` invece di `n_kv_heads`
sovrastimerebbe di un ordine di grandezza) e overhead; se non entra, dice cosa
scaricare e in che ordine, oppure come farlo entrare riducendo il contesto o
quantizzando la cache.

**Ricostruire il passato.** Storico continuo su SQLite con riduzione progressiva
della risoluzione. Quando un processo sparisce mentre la GPU è quasi piena, viene
registrato un incidente con i 60 secondi precedenti: chi teneva cosa, quanto era
piena la GPU, cosa stava crescendo.

---

## Installazione

Nessuna dipendenza da compilare.

```powershell
pip install -r requirements.txt

# servizio con avvio automatico e riavvio in caso di crash (richiede NSSM)
.\install_service.ps1
```

Il servizio gira come utente interattivo, non come `SYSTEM`: i contatori per
processo e le API locali dei backend sono visibili solo in quel contesto.

```powershell
pythonw .\vramexp_gui.py     # interfaccia grafica
python -m vramexp top        # riga di comando
```

La GUI funziona anche senza servizio installato, campionando in proprio; in quel
caso lo storico precedente all'avvio non esiste, e la barra di stato lo dichiara.

---

## Riga di comando

```
vramexp status                       stato della GPU
vramexp top [--limit N] [--actions]  processi per memoria occupata
vramexp watch                        aggiornamento continuo
vramexp actions <pid>                azioni disponibili per un processo
vramexp unload <pid> [--kind K] [--yes]
vramexp reserve 8G                   riserva VRAM (0 per rilasciare)
vramexp policy [--set no-fallback] [--exe python.exe]
vramexp plan <modello> [--context N] [--kv-type q8_0]
vramexp library [query] [--rescan]
vramexp profiles [--preview NOME] [--apply NOME --yes]
vramexp incidents
vramexp probe <pid> [--attach --yes]
```

---

## Introspezione dei processi PyTorch

Modo sicuro, senza rischi per il processo:

```powershell
$env:VRAMEXP_AGENT = "1"
python il_mio_script.py
```

L'agent apre una named pipe locale ed espone, su richiesta, le singole
allocazioni con lo stack Python che le ha create, la frammentazione
(`riservato − allocato`), il numero di retry dell'allocatore e gli OOM.

Esiste anche l'iniezione a caldo in un processo già avviato. È disattivata per
default e va abilitata consapevolmente (`enable_hot_attach`): scrive nella
memoria del processo ed esegue codice al suo interno, e in casi sfortunati può
bloccarlo o farlo terminare. Su un lavoro lungo conviene sempre la variabile
d'ambiente.

---

## API locale

`http://127.0.0.1:8765`, vincolata al loopback, con token in
`%LOCALAPPDATA%\VramExplorer\api_token.txt`.

```
GET  /status /snapshot /processes /process/{pid} /process/{pid}/allocations
GET  /history/gpu /history/processes /incidents /library /plan /policy /reserve
POST /action /reserve /policy /profiles/preview /profiles/apply /probe/attach
WS   /live
```

Le operazioni distruttive richiedono `"confirmed": true` esplicito.

---

## Limiti dichiarati

- **Non si vedono le allocazioni dei processi terzi.** Windows espone solo il
  totale per processo; il dettaglio esiste unicamente dove il processo collabora
  (agent PyTorch o API del backend). Nessuno strumento fa diversamente su WDDM.
- **Non esiste modo di liberare selettivamente la memoria altrui** senza la
  collaborazione del processo.
- **La riserva è best-effort**, non un blocco hardware.
- **`dwm.exe` occupa spesso molti gigabyte**: è memoria di composizione delle
  finestre, in larga parte attribuibile a browser e app accelerate. Non è
  terminabile e lo strumento lo impedisce, spiegando invece come ridurla.
- **La modifica della policy del driver richiede l'elevazione** (verificato:
  `NVAPI_INVALID_USER_PRIVILEGE`, -137, senza UAC). Compare un prompt visibile,
  mai un'elevazione silenziosa.

---

## Struttura

```
vramexp/
  core/      modelli, unità, configurazione
  collect/   PDH, NVML, DXGI, processi, NVAPI, sampler
  content/   riconoscimento del contenuto per backend + header GGUF
  store/     storico SQLite, retention, incidenti
  control/   unload, riserva, policy di fallback, azioni sui processi
  probe/     agent PyTorch, client, iniezione
  planner/   stima cache KV, libreria modelli, what-if, profili
  api/       REST e WebSocket
  cli/       riga di comando
gui/         PySide6: dashboard, timeline, controlli, pianificatore
```
