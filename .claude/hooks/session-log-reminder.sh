#!/bin/bash
TODAY=$(date +%Y-%m-%d)
LOG="/home/samy/Documents/Obsidian Vault/logs/SESSION_LOG.md"
if ! grep -q "$TODAY" "$LOG" 2>/dev/null; then
  echo "SESSION LOG: No entry for $TODAY — run /session-log before ending the session."
fi
