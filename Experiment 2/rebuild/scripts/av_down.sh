#!/usr/bin/env bash
# av_down.sh — stop the NLA AV SGLang server (free VRAM before loading a target model).
set -euo pipefail
if pkill -f "sglang.launch_server"; then
  echo "stopped sglang.launch_server"
else
  echo "no sglang.launch_server process found"
fi
