# Kimi-K2.7-Code — Guida alla configurazione benchmark su MI355X

> Documento di riferimento per la configurazione `kimik2.7-fp4-mi355x-vllm-agentic-lmcache`
> Aggiornato: 2026-06-29

---

## 1. Cos'è il modello

| Campo              | Valore                                         |
|--------------------|------------------------------------------------|
| **HuggingFace ID** | `amd/Kimi-K2.7-Code-MXFP4`                   |
| **Famiglia**       | Kimi K2 — modello MoE (Mixture-of-Experts) di Moonshot AI, specializzato per agentic coding |
| **Quantizzazione** | **MXFP4** (MX-standard FP4 via AMD Quark) — pesi a 4 bit con scaling block per precisione |
| **Predecessore**   | `Kimi-K2.5` (usato in `kimik2.5_fp4_mi355x.sh`) |
| **Framework**      | vLLM 0.21.0 ROCm 7.2.2                        |
| **Hardware**       | AMD MI355X (8× GPU per nodo)                  |

---

## 2. Dove vive la configurazione

```
amd-master.yaml              ← chiave "kimik2.7-fp4-mi355x-vllm-agentic-lmcache"
  │
  ├─ image:        vllm/vllm-openai-rocm:v0.21.0
  ├─ model:        amd/Kimi-K2.7-Code-MXFP4
  ├─ model-prefix: kimik2.7-code    ← determina il nome dello script
  ├─ runner:       mi355x
  ├─ precision:    fp4
  └─ framework:    vllm
```

### Come `model-prefix` seleziona lo script

Il launch script `launch_mi355x-amd.sh` fa:

```bash
MODEL_CODE="${EXP_NAME%%_*}"   # estrae prefisso prima del primo underscore
# cerca: upstream/InferenceX/benchmarks/single_node/agentic/${MODEL_CODE}_fp4_mi355x.sh
```

Quindi `model-prefix: kimik2.7-code` → cerca `kimik2.7-code_fp4_mi355x.sh`.
**Bug risolto**: in origine era `kimik2.7` → cercava `kimik2.7_fp4_mi355x.sh` (inesistente).

---

## 3. Scenari configurati

```yaml
scenarios:
  agentic-coding:
  - duration: 1800   # 30 minuti per run
    search-space:
    - { tp: 8, ep: 1, offloading: none,     conc-list: [16] }
    - { tp: 8, ep: 1, offloading: lmcache,  total-cpu-dram-gb: 2500, conc-list: [16] }
```

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `tp` | 8 | Tensor Parallelism — il modello è shardato su tutti e 8 gli MI355X |
| `ep` | 1 | Expert Parallelism disabilitato (nessun `--enable-expert-parallel`) |
| `conc` | 16 | Numero massimo di sequenze parallele (`--max-num-seqs`) |
| `duration` | 1800 s | Durata della sessione di replay delle trace |
| `offloading` | `none` / `lmcache` | Strategia KV-cache offload (vedi §5) |

---

## 4. Script benchmark — cosa fa

**File**: [`upstream/InferenceX/benchmarks/single_node/agentic/kimik2.7-code_fp4_mi355x.sh`](../upstream/InferenceX/benchmarks/single_node/agentic/kimik2.7-code_fp4_mi355x.sh)

### Flusso esecuzione

```
1. Download modello (hf download amd/Kimi-K2.7-Code-MXFP4)
2. rocm-smi / amd-smi  → stampa stato GPU
3. resolve_trace_source → individua corpus trace agentic
4. install_agentic_deps → installa dipendenze Python per il replay
5. pip install amd-quark   ← necessario per MXFP4 (workaround bug ROCm vLLM)
6. Workarounds hardware:
   - VLLM_ROCM_USE_AITER_RMSNORM=0  se TP < 8  (accuracy issues)
   - HSA_NO_SCRATCH_RECLAIM=1        se firmware MEC < 177 (RCCL memory reclaim bug)
7. Configura offloading (none / cpu / lmcache) → vedi §5
8. Avvia vLLM server in background
9. wait_for_server_ready
10. build_replay_cmd + run_agentic_replay_and_write_outputs
```

### Variabili d'ambiente obbligatorie

| Variabile | Esempio | Note |
|-----------|---------|------|
| `MODEL` | `amd/Kimi-K2.7-Code-MXFP4` | HF model ID |
| `TP` | `8` | Tensor parallel size |
| `CONC` | `16` | Max concurrent sequences |
| `OFFLOADING` | `none` / `lmcache` | Modalità KV offload |
| `TOTAL_CPU_DRAM_GB` | `2500` | Pool CPU DRAM (rilevante per offloading) |
| `RESULT_DIR` | `/path/to/results` | Directory output |
| `EP_SIZE` | `1` | Expert parallel size |
| `DP_ATTENTION` | `false` | Data-parallel attention |

### Comando vLLM generato (caso `none`)

```bash
vllm serve $MODEL_PATH \
  --served-model-name $MODEL \
  --host 0.0.0.0 --port 8888 \
  --tensor-parallel-size=8 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --block-size=1 \
  --trust-remote-code \
  --max-num-seqs 16 \
  --mm-encoder-tp-mode data
```

Punti chiave:
- `--kv-cache-dtype fp8` — KV cache compressa in FP8 (risparmio ~50% vs BF16)
- `--block-size=1` — granularità minima per paged KV (richiesta da LMCache)
- `--mm-encoder-tp-mode data` — encoder multimodale in data-parallel mode

---

## 5. Modalità offloading KV

### 5a. `none` — solo GPU KV

Nessun offload. KV cache risiede interamente nella HBM degli MI355X (~1.5 TB totale su 8 GPU).

### 5b. `cpu` — vLLM native CPU offload

```bash
--kv_offloading_backend native
--kv_offloading_size $TOTAL_CPU_DRAM_PARTITION_GB   # ~375 GB per rank (3000/8)
--disable-hybrid-kv-cache-manager
```

- Usa `OffloadingConnector` interno a vLLM (NON `SimpleCPUOffloadConnector`)
- Pool hardcodato a 3000 GB totali (override del parametro `total-cpu-dram-gb`)
- Partizionato automaticamente per rank TP: `3000 GB / (8 / TP)`

### 5c. `lmcache` — LMCache MP server

Architettura a due processi:

```
┌─────────────────────────┐     ZMQ :5555      ┌──────────────────────────┐
│   vLLM serve (8× GPU)   │ ←─────────────────→ │   LMCache MP server      │
│   LMCacheMPConnector     │                    │   CPU DRAM pool: 3 TB    │
│   kv_role: kv_both       │     HTTP :8080     │   healthcheck endpoint   │
└─────────────────────────┘ ←─── healthcheck ── └──────────────────────────┘
```

Parametri LMCache:
| Parametro | Default | Significato |
|-----------|---------|-------------|
| `LMCACHE_L1_SIZE_GB` | 3000 | Pool totale CPU KV |
| `LMCACHE_L1_INIT_SIZE_GB` | 20 | Allocazione iniziale (crescita lazy) |
| `LMCACHE_L1_READ_TTL_SECONDS` | 7200 | Lease di lettura (2h, esteso perché TP8/conc16 può impiegare >300s tra lookup e retrieve) |
| `LMCACHE_CHUNK_SIZE` | 256 | Token per chunk KV |
| `LMCACHE_MAX_WORKERS` | `$TP` (=8) | Thread worker ZMQ |
| `LMCACHE_BLOCKING_TIMEOUT_SECS` | 120 | Timeout blocco singola operazione |
| Eviction policy | LRU | Least Recently Used |

Argomenti vLLM per LMCache:
```bash
--enable-prefix-caching
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector",
                        "kv_role":"kv_both",
                        "kv_connector_extra_config":{
                          "lmcache.mp.host":"tcp://127.0.0.1",
                          "lmcache.mp.port":5555}}'
--disable-hybrid-kv-cache-manager
```

---

## 6. Differenze rispetto a Kimi-K2.5

| Aspetto | K2.5 (`kimik2.5_fp4_mi355x.sh`) | K2.7-Code (`kimik2.7-code_fp4_mi355x.sh`) |
|---------|----------------------------------|-------------------------------------------|
| Modello HF | (non specificato qui) | `amd/Kimi-K2.7-Code-MXFP4` |
| LMCache install | `git clone HEAD` (non pinnato) | `git fetch --depth 1 origin <sha>` pinnato |
| Commit LMCache | HEAD (variabile) | `4bbfd11b` (30 giugno 2026 ≈ ultimo compatibile con vLLM 0.21) |
| Setup iniziale | Uguale | + `echo [lmcache] Active commit:` per debugging |
| vLLM command | Identico | Identico |

---

## 7. Problemi noti e fix applicati

### 7.1 `No such file or directory: kimik2.7_fp4_mi355x.sh`
- **Causa**: `model-prefix: kimik2.7` → cercava script con nome sbagliato
- **Fix**: `model-prefix: kimik2.7-code` in `amd-master.yaml` (riga 2217)

### 7.2 `ImportError: cannot import name 'KVCacheSpecKind'`
- **Causa**: LMCache commit `65c2ae8` (11 giugno 2026) ha aggiunto `kv_cache_group_edits.py` che importa `KVCacheSpecKind`, non presente in vLLM 0.21.0
- **Fix**: pin a commit `4bbfd11b` via `git fetch --depth 1 origin <sha>` + `git checkout FETCH_HEAD`
- **Perché non bastava `git checkout`**: il runner CI usa clone parziale, checkout senza fetch falliva silenziosamente

### 7.3 `ConnectionError: LMCache server did not respond to register_kv_caches within 300.0s`
- **Stato**: in analisi — necessario `lmcache_server.log` dall'artifact GitHub Actions
- **Sintomi**: HTTP healthcheck OK, ma i worker vLLM (TP0-TP7) non ricevono risposta ZMQ alla chiamata `register_kv_caches`
- **Possibili cause**: incompatibilità protocollo ZMQ con vLLM 0.21 al commit `4bbfd11b`, deadlock interno LMCache, porta 5555 occupata

---

## 8. Come lanciare

### Via `gg_agentic` (metodo locale)

```bash
# Dispatch singolo + monitoring
python gg_agentic/run_and_watch.py

# Oppure con lo script dedicato
bash gg_agentic/launch_kimik27.sh [--ref BRANCH] [--force] [--dry-run]
```

`run_config.yaml` configurato con:
```yaml
config-keys: kimik2.7-fp4-mi355x-vllm-agentic-lmcache
runner: mi355x
```

### Via GitHub Actions (diretto)

```bash
gh workflow run e2e-tests.yml \
  --repo ROCm/InferenceMAX_rocm \
  --ref chore/agentx-v0.4 \
  -f "generate-cli-command=test-config --config-files .github/configs/amd-master.yaml \
      --config-keys kimik2.7-fp4-mi355x-vllm-agentic-lmcache"
```

---

## 9. Artifact di output

Dopo ogni run, GitHub Actions carica in "Upload agentic raw results":

| File | Contenuto |
|------|-----------|
| `server.log` | Log vLLM serve (errori startup, GPU alloc, request log) |
| `lmcache_server.log` | Log LMCache MP server (ZMQ bind, register_kv_caches, errori) |
| `lmcache_command.txt` | Comando LMCache esatto usato nella run |
| `vllm_command.txt` | Comando vLLM esatto usato nella run |
| `*.jsonl` / `*.csv` | Risultati benchmark (throughput, latency, TTFT) |

---

## 10. Env var utili per debug/override

```bash
# Sovrascrivere il commit LMCache pinned
export LMCACHE_REF=<altra-sha>

# Aumentare il timeout healthcheck LMCache (default 120 tentativi × 1s)
export LMCACHE_READY_ATTEMPTS=300

# Forzare dimensione pool CPU
export LMCACHE_L1_SIZE_GB=2000

# Disabilitare AITER esplicitamente
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_RMSNORM=0

# Porta alternativa per vLLM
export PORT=9999
```
