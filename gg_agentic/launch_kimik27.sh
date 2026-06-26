#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_kimik27.sh
#
# Verifica che nessun benchmark stia girando sui runner mi355x (proxy per
# "porta 8888 libera"), poi dispatcha e2e-tests.yml con:
#   kimik2.7-fp4-mi355x-vllm-agentic-lmcache
#
# Uso:
#   bash gg_agentic/launch_kimik27.sh [--ref BRANCH] [--force] [--dry-run]
#
# Opzioni:
#   --ref BRANCH   branch/sha su cui girare (default: branch corrente)
#   --force        lancia anche se ci sono run attive sui runner mi355x
#   --dry-run      stampa cosa farebbe senza dispatchare
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
WORKFLOW="e2e-tests.yml"
CONFIG_KEY="kimik2.7-fp4-mi355x-vllm-agentic-lmcache"
CONFIG_FILES=".github/configs/amd-master.yaml"
RUNNER_LABEL="mi355x"
REPO="ROCm/InferenceMAX_rocm"

# ---------------------------------------------------------------------------
# Parsing argomenti
# ---------------------------------------------------------------------------
REF=""
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)      REF="$2"; shift 2 ;;
        --force)    FORCE=true; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        *)          echo "Argomento sconosciuto: $1"; exit 1 ;;
    esac
done

# Branch di default = branch corrente del repo locale
if [[ -z "$REF" ]]; then
    REF=$(git -C "$(dirname "$0")/.." rev-parse --abbrev-ref HEAD 2>/dev/null || echo "chore/agentx-v0.4")
fi

# ---------------------------------------------------------------------------
# Autenticazione: usa GH_TOKEN se settato, altrimenti la sessione gh CLI.
# Evita di usare GITHUB_TOKEN (classic PAT) che ROCm blocca.
# ---------------------------------------------------------------------------
if [[ -n "${GH_TOKEN:-}" ]]; then
    export GH_TOKEN
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
    # Rimappa su GH_TOKEN in modo che gh lo usi come fine-grained
    export GH_TOKEN="${GITHUB_TOKEN}"
    unset GITHUB_TOKEN          # evita che gh usi il classic bloccato
fi

# Verifica che gh sia autenticato
if ! gh auth status &>/dev/null; then
    echo "❌  gh CLI non autenticato. Esegui: gh auth login"
    exit 1
fi

# ---------------------------------------------------------------------------
# Funzione: controlla run attive sui runner mi355x
# Ritorna 0 se il runner e' libero, 1 se occupato.
# ---------------------------------------------------------------------------
check_runner_busy() {
    echo "🔍  Controllo run attive su runner '$RUNNER_LABEL' ..."

    # Ottieni le run in_progress o queued
    local active_runs
    active_runs=$(gh run list \
        --repo "$REPO" \
        --workflow "$WORKFLOW" \
        --status in_progress \
        --json databaseId,displayTitle,status \
        --limit 20 \
        2>/dev/null || echo "[]")

    local queued_runs
    queued_runs=$(gh run list \
        --repo "$REPO" \
        --workflow "$WORKFLOW" \
        --status queued \
        --json databaseId,displayTitle,status \
        --limit 20 \
        2>/dev/null || echo "[]")

    # Combina e itera su ogni run attiva per cercare job su mi355x
    local busy_runs=()
    local all_ids
    all_ids=$(echo "$active_runs $queued_runs" \
        | python3 -c "
import sys, json
a = []
for chunk in sys.stdin.read().split('] ['):
    chunk = chunk.strip().strip('[]')
    if chunk:
        try: a += json.loads('[' + chunk + ']')
        except: pass
print('\n'.join(str(r['databaseId']) for r in a))
" 2>/dev/null || true)

    if [[ -z "$all_ids" ]]; then
        echo "✅  Nessuna run attiva su $WORKFLOW."
        return 0
    fi

    local found_mi355x=false
    while IFS= read -r run_id; do
        [[ -z "$run_id" ]] && continue

        # Controlla i job di questa run
        local jobs_json
        jobs_json=$(gh run view "$run_id" \
            --repo "$REPO" \
            --json jobs \
            2>/dev/null || echo '{"jobs":[]}')

        local mi355x_jobs
        mi355x_jobs=$(echo "$jobs_json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
hits = [j['name'] for j in d.get('jobs', [])
        if '$RUNNER_LABEL' in j.get('labels', []) or '$RUNNER_LABEL' in j.get('name','').lower()
        and j.get('status') in ('in_progress','queued','waiting')]
print('\n'.join(hits))
" 2>/dev/null || true)

        if [[ -n "$mi355x_jobs" ]]; then
            echo "⚠️   Run #$run_id ha job attivi su '$RUNNER_LABEL':"
            echo "$mi355x_jobs" | sed 's/^/      - /'
            found_mi355x=true
        fi
    done <<< "$all_ids"

    if $found_mi355x; then
        return 1
    fi

    echo "✅  Runner '$RUNNER_LABEL' sembra libero (nessun job attivo trovato)."
    return 0
}

# ---------------------------------------------------------------------------
# Funzione: dispatcha il workflow
# ---------------------------------------------------------------------------
dispatch_workflow() {
    local cmd=(
        gh workflow run "$WORKFLOW"
        --repo "$REPO"
        --ref "$REF"
        -f "generate-cli-command=test-config --config-files $CONFIG_FILES --config-keys $CONFIG_KEY"
    )

    echo ""
    echo "🚀  Dispatch:"
    echo "     workflow : $WORKFLOW"
    echo "     ref      : $REF"
    echo "     config   : $CONFIG_KEY"
    echo ""

    if $DRY_RUN; then
        echo "[dry-run] Comando che verrebbe eseguito:"
        echo "  ${cmd[*]}"
        return 0
    fi

    "${cmd[@]}"
    echo ""
    echo "✅  Workflow dispatchato. Controlla lo stato con:"
    echo "     gh run list --repo $REPO --workflow $WORKFLOW --limit 5"
    echo ""

    # Aspetta qualche secondo e mostra la run appena creata
    sleep 4
    gh run list --repo "$REPO" --workflow "$WORKFLOW" --limit 3
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "========================================"
echo " launch_kimik27.sh"
echo " Repo   : $REPO"
echo " Ref    : $REF"
echo " Config : $CONFIG_KEY"
echo "========================================"
echo ""

if ! check_runner_busy; then
    if $FORCE; then
        echo "⚠️   Runner occupato ma --force specificato. Procedo comunque."
    else
        echo ""
        echo "❌  Runner '$RUNNER_LABEL' e' occupato — porta 8888 probabilmente gia' in uso."
        echo "    Aspetta che le run attive finiscano, oppure usa --force per ignorare."
        exit 1
    fi
fi

dispatch_workflow
