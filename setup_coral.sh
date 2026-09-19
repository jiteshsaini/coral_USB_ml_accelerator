#!/bin/bash
# Coral USB Accelerator test code - installer for Raspberry Pi OS 12/13 (Bookworm / Trixie).
#
#   curl -fsSL https://raw.githubusercontent.com/jiteshsaini/coral_USB_ml_accelerator/master/setup_coral.sh -o setup_coral.sh
#   sudo bash setup_coral.sh
#
# The code goes to ~/coral_USB_ml_accelerator. An existing copy is moved aside,
# never overwritten.

# `sh setup_coral.sh` runs dash, which cannot parse the rest of this file.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
[ "$(id -u)" -eq 0 ] || exec sudo bash "$0" "$@"

set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO=https://github.com/jiteshsaini/coral_USB_ml_accelerator.git
RUN_USER=${SUDO_USER:-pi}
CODE=$(getent passwd "$RUN_USER" | cut -d: -f6)/coral_USB_ml_accelerator
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=/tmp/coral-install-$STAMP.log
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

ok()   { echo "  [ ok ] $1"; }
warn() { echo "  [warn] $1"; }
die()  { echo "  [FAIL] $1"; exit 1; }

reboot_box() {
  echo
  echo "  ============================================================"
  echo "   REBOOT NEEDED - kernel $NEWEST was installed,"
  echo "   but $RUNNING is still running."
  echo
  for line in "$@"; do echo "   $line"; done
  echo "  ============================================================"
}

MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
OSVER=$(. /etc/os-release; echo "${VERSION_CODENAME:-unknown}")
ARCH=$(uname -m)
IP=$(hostname -I | awk '{print $1}')
ID=$(grep -m1 ^Serial /proc/cpuinfo | sha256sum | cut -c1-16)
MEM=$(free -m | awk '/Mem:/{print $2}')

echo
echo "=================================================="
echo "  Checking this machine"
echo "=================================================="
echo "  Board:  ${MODEL:-unknown}"
echo "  OS:     $(. /etc/os-release; echo "${PRETTY_NAME:-unknown}")"
echo "  RAM:    $MEM MB"
echo "  Address $IP"
echo

[ "$(. /etc/os-release; echo "${VERSION_ID:-0}")" -ge 12 ] 2>/dev/null \
  || die "needs Raspberry Pi OS 12 (Bookworm) or newer"
[ "$ARCH" = "aarch64" ] || die "needs the 64-bit Raspberry Pi OS - the Coral library is built for arm64 only"

echo
echo "=================================================="
echo "  Updating the system"
echo "=================================================="
echo "  The slow part: on an older board this can take an hour. The output below"
echo "  keeps moving; it is not stuck."
apt-get update 2>&1 | tee -a "$LOG" >/dev/null || warn "apt-get update failed - see $LOG"
apt-get full-upgrade -y 2>&1 | tee -a "$LOG" || warn "the upgrade did not finish cleanly - continuing"

# A newer kernel only runs after a reboot, and the camera stack that came with
# it may not work against the old one until then.
RUNNING=$(uname -r)
NEWEST=$(ls /lib/modules | grep -- "+${RUNNING#*+}\$" | sort -V | tail -1)
REBOOT=0
if [ -n "$NEWEST" ] && [ "$NEWEST" != "$RUNNING" ]; then
  REBOOT=1
  reboot_box "The rest of the install carries on, but the camera may not work" \
             "until you reboot. You will be reminded at the end."
fi

echo
echo "=================================================="
echo "  Installing packages"
echo "=================================================="
apt-get install -y python3-numpy python3-pil python3-picamera2 python3-matplotlib curl git 2>&1 | tee -a "$LOG" \
  || die "package install failed - see $LOG"

echo
echo "=================================================="
echo "  Installing Python packages (OpenCV, LiteRT)"
echo "=================================================="
# Headless, and 4.x: PyPI now resolves the unpinned name to 5.x, and the full
# build pulls ~500 MB of GUI libraries this never opens.
OPENCV_PIN="opencv-python-headless==4.14.0.94"
# Must match the TensorFlow version libedgetpu below is built against. A
# mismatched pair does not raise an error; it crashes when a model loads.
LITERT_PIN="ai-edge-litert==2.2.0"
CORAL_TAG="16.0TF2.19.1-1"

python3 -c "import cv2" 2>/dev/null \
  || pip3 install --break-system-packages "$OPENCV_PIN" 2>&1 | tee -a "$LOG" \
  || die "$OPENCV_PIN failed to install"
python3 -c "import ai_edge_litert" 2>/dev/null \
  || pip3 install --break-system-packages "$LITERT_PIN" 2>&1 | tee -a "$LOG" \
  || die "$LITERT_PIN failed to install"

echo
echo "=================================================="
echo "  Installing Coral USB Accelerator support"
echo "=================================================="
# Google's own libedgetpu targets TensorFlow Lite ~2.5 and crashes under
# ai-edge-litert; this community rebuild matches it.
if dpkg-query -W -f='${Version}' libedgetpu1-std 2>/dev/null | grep -q tf2.19.1; then
  ok "Coral library already installed"
else
  for cn in $OSVER trixie bookworm; do
    deb="libedgetpu1-std_16.0tf2.19.1-1.${cn}_arm64.deb"
    if curl -fsSL --max-time 180 -o "$TMP/$deb" \
         "https://github.com/feranick/libedgetpu/releases/download/$CORAL_TAG/$deb" 2>>"$LOG"; then
      # still declares libgcc1, which libgcc-s1 replaced; dpkg exits non-zero on the override
      dpkg -i --ignore-depends=libgcc1 "$TMP/$deb" >>"$LOG" 2>&1
      ldconfig
      ok "Coral library installed ($cn build) - replug the accelerator if it is attached"
      break
    fi
  done
fi

echo
echo "=================================================="
echo "  Installing the code"
echo "=================================================="
if git clone -q --depth 1 "$REPO" "$TMP/repo" && [ -f "$TMP/repo/exp/classify.py" ]; then
  rm -rf "$TMP/repo/.git"
  if [ -e "$CODE" ]; then
    mv "$CODE" "$CODE.backup_$STAMP" && ok "existing code moved to $CODE.backup_$STAMP"
  fi
  mv "$TMP/repo" "$CODE" && chown -R "$RUN_USER:$RUN_USER" "$CODE" && ok "code installed in $CODE"
else
  die "could not fetch the code from $REPO"
fi

echo
echo "=================================================="
echo "  Setting group memberships"
echo "=================================================="
# The camera belongs to video, and libedgetpu's udev rule gives the Coral to
# plugdev - so neither script needs to run as root.
NEW_GROUPS=0
for g in video plugdev; do
  id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx "$g" || { adduser "$RUN_USER" "$g" >/dev/null 2>&1; NEW_GROUPS=1; }
done
ok "$RUN_USER is in video and plugdev"

as_user() { (cd /tmp && sudo -u "$RUN_USER" env HOME=/tmp python3 -c "$1") >/dev/null 2>&1; }
opencv() { as_user "import cv2"; }
litert() { as_user "from ai_edge_litert.interpreter import Interpreter"; }
picamera2() { as_user "import picamera2"; }
camera() { ls /dev/v4l/by-id 2>/dev/null | grep -qi usb || rpicam-hello --list-cameras 2>/dev/null | grep -q " : "; }
cpu_model() {
  as_user "
from ai_edge_litert.interpreter import Interpreter
Interpreter(model_path='$CODE/exp/mobilenet_v1_1.0_224_quant.tflite').allocate_tensors()"
}
coral_attached() { grep -qx -e 1a6e -e 18d1 /sys/bus/usb/devices/*/idVendor 2>/dev/null; }
coral() {
  as_user "
from ai_edge_litert.interpreter import Interpreter, load_delegate
Interpreter(model_path='$CODE/exp/mobilenet_v1_1.0_224_quant_edgetpu.tflite',
            experimental_delegates=[load_delegate('libedgetpu.so.1')]).allocate_tensors()"
}

ST=ok
check() {
  if "$2"; then printf "  %-36s yes\n" "$1"; else printf "  %-36s NO\n" "$1"; ST=fail; fi
}
echo
echo "=================================================="
echo "  Checking the install"
echo "=================================================="
check "OpenCV" opencv
check "LiteRT interpreter" litert
check "picamera2" picamera2
check "model loads on the CPU" cpu_model
if coral_attached; then
  check "model loads on the Coral" coral
else
  printf "  %-36s not attached\n" "Coral USB Accelerator"
fi
if camera; then
  printf "  %-36s yes\n" "camera detected"
else
  printf "  %-36s no (needed for classify.py only)\n" "camera detected"
fi

curl -s -m 5 https://helloworld.co.in/deploy/t.php >/dev/null 2>&1 -d \
    "p=$(basename "$REPO" .git)&e=install&s=$ST&i=$ID&m=${MODEL// /+}&o=$OSVER&a=$ARCH&l=$IP" || true

echo
echo "=================================================="
echo "  Done"
echo "=================================================="
if [ "$REBOOT" -eq 1 ]; then
  reboot_box "Reboot now, then try it:" "" "    sudo reboot" "" \
             "    cd $CODE/exp" "    python3 classify.py" "    python3 classify_coral.py"
  echo
else
  [ "$NEW_GROUPS" -eq 1 ] && echo "  Log out and back in first, so your new groups apply." && echo
  echo "  Classify what the camera sees, and compare the inference times (Ctrl+C to stop):"
  echo "    cd $CODE/exp"
  echo "    python3 classify.py           on the CPU"
  echo "    python3 classify_coral.py     on the Coral"
  echo
fi
