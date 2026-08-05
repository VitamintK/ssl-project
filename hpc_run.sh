#!/usr/bin/env bash
# Sync local code to the HPC and submit a SLURM job.
#
# Usage:
#   ./hpc_run.sh [options] -- COMMAND
#
# Options:
#   --gpus N            Number of GPUs (default: 1)
#   --time HH:MM:SS     Wall time limit (default: 24:00:00)
#   --partition NAME    SLURM partition (default: gpu)
#   --mem MEM           Memory request (default: 32G)
#   --no-sync           Skip the rsync step
#   --no-neupl          Skip syncing NEUPL checkpoint directories
#
# Example:
#   ./hpc_run.sh --gpus 1 --time 04:00:00 -- uv run python eval_init_diversity.py --source psro

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

GPUS=1
TIME="24:00:00"
# PARTITION="gpu"
PARTITION='3090-gcondo'
MEM="32G"
SYNC=true
NEUPL_SYNC=true

SSH_KEY="$HOME/.ssh/id_rsa_brown"
SSH_HOST="kawang@sshcampus.ccv.brown.edu"
REMOTE_DIR="projects/ssl-project"       # relative to home
REMOTE_BENCHMARK_DIR="projects/IIG-RL-Benchmark"  # relative to home
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: $0 [--gpus N] [--time HH:MM:SS] [--partition NAME] [--mem MEM] [--no-sync] -- COMMAND"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)      GPUS="$2";      shift 2 ;;
        --time)      TIME="$2";      shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --mem)       MEM="$2";       shift 2 ;;
        --no-sync)   SYNC=false;     shift   ;;
        --no-neupl)  NEUPL_SYNC=false; shift  ;;
        --)          shift; break ;;
        -*) usage ;;
        *)  break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: no command specified."
    usage
fi

CMD="$*"

# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

if [[ "$SYNC" == true ]]; then
    echo "==> Syncing ${LOCAL_DIR}/ to ${SSH_HOST}:~/${REMOTE_DIR}/ ..."
    rsync -avz --exclude='.git' \
               --exclude='__pycache__' \
               --exclude='*.pyc' \
               --exclude='.venv' \
               --exclude='results/' \
               --exclude='logs/' \
        -e "ssh -i ${SSH_KEY}" \
        "${LOCAL_DIR}/" "${SSH_HOST}:~/${REMOTE_DIR}/"
    echo "==> Sync complete."

    if [[ "$NEUPL_SYNC" == true ]]; then
        NEUPL_BASE="results/test/neupl/ppo/hs256"
        for game in kuhn_poker leduc_poker; do
            GAME_DIR="${LOCAL_DIR}/${NEUPL_BASE}/${game}"
            [[ -d "$GAME_DIR" ]] || continue

            NEUPL_DIRS=()
            while IFS= read -r line; do
                [[ -n "$line" ]] && NEUPL_DIRS+=("$line")
            done < <(ls -dt "${GAME_DIR}"/*/  2>/dev/null | sed 's|/$||' | head -10)
            [[ ${#NEUPL_DIRS[@]} -gt 0 ]] || continue

            echo ""
            echo "==> NEUPL directories for ${game} (most recent first):"
            for i in "${!NEUPL_DIRS[@]}"; do
                echo "  $((i+1))) $(basename "${NEUPL_DIRS[$i]}")"
            done
            echo -n "  Enter space-separated numbers to sync (or press Enter to skip): "
            read -r selection

            for num in $selection; do
                if [[ "$num" =~ ^[0-9]+$ ]] && (( num >= 1 && num <= ${#NEUPL_DIRS[@]} )); then
                    name="$(basename "${NEUPL_DIRS[$((num-1))]}")"
                    echo "==> Syncing ${game}/${name}"
                    ssh -i "${SSH_KEY}" "${SSH_HOST}" \
                        "mkdir -p ~/${REMOTE_DIR}/${NEUPL_BASE}/${game}/${name}"
                    rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
                        -e "ssh -i ${SSH_KEY}" \
                        "${GAME_DIR}/${name}/" \
                        "${SSH_HOST}:~/${REMOTE_DIR}/${NEUPL_BASE}/${game}/${name}/"
                fi
            done
        done
    fi

    echo "==> Updating IIG-RL-Benchmark on HPC..."
    ssh -i "${SSH_KEY}" "${SSH_HOST}" "git -C ~/${REMOTE_BENCHMARK_DIR} pull && cd ~/${REMOTE_DIR} && [ -d .venv ] || uv venv --python 3.12 && uv pip install -r requirements.txt && uv pip install -e ~/${REMOTE_BENCHMARK_DIR}"
    echo "==> IIG-RL-Benchmark up to date."
fi

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

echo "==> Submitting job: ${CMD}"

SBATCH_OUT=$(ssh -i "${SSH_KEY}" "${SSH_HOST}" \
    "mkdir -p ~/${REMOTE_DIR}/logs && sbatch \
        --output=\$HOME/${REMOTE_DIR}/logs/slurm-%j.out \
        --error=\$HOME/${REMOTE_DIR}/logs/slurm-%j.err \
        --chdir=\$HOME/${REMOTE_DIR}" <<EOF
#!/bin/bash
#SBATCH --job-name=ssl-project
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPUS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}

${CMD}
EOF
)

echo "==> ${SBATCH_OUT}"

JOB_ID=$(echo "${SBATCH_OUT}" | grep -oE '[0-9]+')
if [[ -z "${JOB_ID}" ]]; then
    echo "Error: could not parse job ID from sbatch output."
    exit 1
fi

LOG_OUT="~/${REMOTE_DIR}/logs/slurm-${JOB_ID}.out"
LOG_ERR="~/${REMOTE_DIR}/logs/slurm-${JOB_ID}.err"
echo "==> Waiting for log files ..."
ssh -i "${SSH_KEY}" "${SSH_HOST}" "until [ -f ${LOG_OUT} ] && [ -f ${LOG_ERR} ]; do sleep 2; done"

echo "==> Streaming stdout and stderr (Ctrl-C to detach):"
ssh -i "${SSH_KEY}" "${SSH_HOST}" "tail -f ${LOG_OUT} ${LOG_ERR}"
