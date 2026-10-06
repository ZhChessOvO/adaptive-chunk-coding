#!/usr/bin/env bash
# Resource guard outside the sender's immutable training protocol.
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
cd "$repo"
mode=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    smoke|train|verify) ;;
    *) echo 'Expected smoke, train or verify.' >&2; exit 2 ;;
esac
exec python -m tools.storage_guard \
    --log "/root/autodl-tmp/DCVC/tmp/routervc_sender_${mode}_storage_guard.jsonl" \
    --min-inodes 5000 -- bash demo/run_routervc_sender.sh "$mode" "$@"
