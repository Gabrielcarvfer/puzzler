#!/usr/bin/env bash
# Live mirror of the phone screen (PuzzleMaster overlay included) in a window on this computer.
# Usage: scripts/watch-phone.sh     (close the window or Ctrl+C to stop)
ADB="$HOME/Android/Sdk/platform-tools/adb"
DEV=$("$ADB" devices | awk 'NR>1 && $2=="device"{print $1; exit}')
[ -z "$DEV" ] && { echo "No phone connected (adb devices is empty)."; exit 1; }
echo "Mirroring $DEV. Close the window to stop."
# screenrecord stops after 3 minutes, so restart it until the viewer window is closed.
while true; do
  "$ADB" -s "$DEV" exec-out screenrecord --output-format=h264 --size 720x1600 --bit-rate 6000000 - 2>/dev/null \
    | ffplay -loglevel error -window_title "PuzzleMaster live" -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 -sync ext -x 540 -y 1200 - 2>/dev/null
  rc=${PIPESTATUS[1]}
  # ffplay exits 0 when its input ends (3-minute limit) and non-zero / by signal when closed by the user.
  sleep 0.5
  "$ADB" -s "$DEV" get-state >/dev/null 2>&1 || { echo "Phone disconnected."; exit 1; }
  [ "$rc" != "0" ] && exit 0
done
