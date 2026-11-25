#!/usr/bin/env bash
# trace one chameleon factorization (fp32) in the gchill-chameleon image and save starpu's tasks.rec
# usage, from the repo root: docker/chameleon/trace.sh potrf 4096 512 data/chameleon/cholesky_8x512.rec
set -euo pipefail
op=$1 n=$2 nb=$3 out=$4
docker run --rm gchill-chameleon bash -c "
  export STARPU_FXT_TRACE=1 STARPU_FXT_PREFIX=/work/ STARPU_SILENT=1
  chameleon_stesting -o s$op -n $n -b $nb -t 4 -T -w > /dev/null
  cd /work && starpu_fxt_tool -i prof_file_* > /dev/null && cat tasks.rec" > "$out"
echo "$out: $(grep -c '^Model' "$out") kernels"