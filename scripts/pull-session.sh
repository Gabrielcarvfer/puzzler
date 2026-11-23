#!/usr/bin/env bash
# Copies everything the phone has scanned to this computer, for offline tests:
#   session/photos/     every photo the app used, each with its sensor data (.txt)
#   session/phone/catalogue.bin   the phone's current catalogue
# Photos already copied are skipped. Usage: scripts/pull-session.sh
set -e
cd "$(dirname "$0")/.."
ADB="$HOME/Android/Sdk/platform-tools/adb"
DEV=$("$ADB" devices | awk 'NR>1 && $2=="device"{print $1; exit}')
[ -z "$DEV" ] && { echo "No phone connected."; exit 1; }
SRC=/sdcard/Android/data/puzzlemaster.app/files/session
mkdir -p session/photos session/phone
n=0
for f in $("$ADB" -s "$DEV" shell ls "$SRC" 2>/dev/null | tr -d '\r'); do
  [ -f "session/photos/$f" ] && continue
  "$ADB" -s "$DEV" pull "$SRC/$f" "session/photos/$f" >/dev/null 2>&1 && n=$((n+1))
done
"$ADB" -s "$DEV" exec-out run-as puzzlemaster.app cat files/catalogue.bin > session/phone/catalogue.bin 2>/dev/null || true
echo "Pulled $n new files. Photos on this computer: $(ls session/photos/*.jpg 2>/dev/null | wc -l). Catalogue: $(stat -c %s session/phone/catalogue.bin 2>/dev/null || echo 0) bytes."
