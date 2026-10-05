#!/usr/bin/env bash
# A storage guard outside the immutable research code/protocol.
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
cd "$repo"
mode=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    train) command=(bash demo/run_routervc_receiver.sh train "$@") ;;
    eval) command=(bash demo/run_routervc_receiver_evaluate.sh run "$@") ;;
    report) command=(bash demo/run_routervc_receiver_evaluate.sh report "$@") ;;
    verify) command=(bash demo/run_routervc_receiver.sh verify "$@") ;;
    *) echo 'Expected train, eval, report or verify.' >&2; exit 2 ;;
esac
exec python -m tools.storage_guard \
    --log "/root/autodl-tmp/DCVC/tmp/routervc_receiver_${mode}_storage_guard.jsonl" \
    --min-inodes 5000 -- "${command[@]}"
