#!/usr/bin/env bash
# The comparison: openodke and two competitors on every prepared set under $CMP,
# then openodke's grounding and corroboration on each competitor's triples.
#   CMP=/path/to/runs/cmp bench/run_all.sh
set -u
cd "$(dirname "$0")/.."
[ -f /root/.config/odke-bench.env ] && { set -a; . /root/.config/odke-bench.env; set +a; }
CMP=${CMP:?set CMP to the directory holding t2k/ont_* and redocred}
dataset () { case "$1" in *redocred*) echo redocred ;; *) echo text2kgbench ;; esac; }
odke_run () {
  local d=$1 start rc
  start=$(date +%s)
  .venv/bin/odke bench run "$(dataset "$d")" "$d" --json > "$d/report.json" 2> "$d/run.log"; rc=$?
  echo "$(date +%T) odke  $(basename "$(dirname "$d")")/$(basename "$d") rc=$rc $(( $(date +%s) - start ))s"
}
extract () {
  local system=$1 d=$2 rc
  bench/.venv/bin/python -W ignore bench/competitors.py extract "$system" "$d" > "$d/extract-$system.log" 2>&1; rc=$?
  echo "$(date +%T) $system extract $(basename "$d") rc=$rc: $(tail -1 "$d/extract-$system.log")"
}
export -f dataset odke_run extract
sets=$(ls -d "$CMP"/t2k/ont_* "$CMP"/redocred)

# 1. openodke itself, and both competitors' extraction, side by side
printf '%s\n' $sets | xargs -P 3 -I{} bash -c 'odke_run "$@"' _ {} &
( for d in $sets; do extract lgt "$d"; done ) &
( for d in $sets; do extract neo4j "$d"; done ) &
wait
# 2. openodke's verification on each competitor's triples
ls -d "$CMP"/t2k/ont_*/competitors/* "$CMP"/redocred/competitors/* | xargs -P 4 -I{} bash -c 'odke_run "$@"' _ {}
echo "ALL DONE $(date +%T)"
