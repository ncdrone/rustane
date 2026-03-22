#!/bin/bash
# monitor-v3.sh — Live dashboard for V3 auto-optimization
#
# Usage: ./system/monitor-v3.sh
#        ./system/monitor-v3.sh beta     # monitor agent beta
# Or in tmux: tmux split-window './system/monitor-v3.sh'

AGENT_ID="${1:-alpha}"
STATUSFILE="/tmp/rustane-v3-status-${AGENT_ID}"
LOGFILE="/tmp/rustane-v3-opt-${AGENT_ID}.log"
WORKTREE="/Users/dan/Dev/rustane-v3-auto-${AGENT_ID}"
EXPERIMENTS="${WORKTREE}/system/experiments-v3.tsv"

watch -n 15 "
echo '=== DeepSeek-V3 Auto-Optimize Dashboard ==='
echo ''
echo 'Agent: ${AGENT_ID}'
echo 'Status:' \$(cat ${STATUSFILE} 2>/dev/null || echo 'offline')
echo ''

if pgrep -f 'optimize-v3.sh' > /dev/null 2>&1; then
    echo 'Loop: RUNNING'
else
    echo 'Loop: NOT RUNNING'
fi
if pgrep -f 'claude.*dangerously-skip-permissions' > /dev/null 2>&1; then
    echo 'Claude: ACTIVE'
else
    echo 'Claude: idle'
fi
echo ''

echo '--- Last 5 Experiments ---'
tail -5 ${EXPERIMENTS} 2>/dev/null | awk -F'\t' '{printf \"%-12s %-20s %-10s %s\n\", \$1, \$2, \$6, \$7}' || echo '(none)'
echo ''

echo '--- Recent Log (last 15 lines) ---'
tail -15 ${LOGFILE} 2>/dev/null || echo '(no log)'
"
