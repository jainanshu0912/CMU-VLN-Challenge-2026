#!/bin/bash
set -e

TARGET_UID="${HOST_UID:-1000}"
TARGET_GID="${HOST_GID:-1000}"

if [ "$(id -u)" -ne 0 ]; then
  exec "$@"
fi

# Open traverse only; do not chown bind mounts (remaps UIDs on the host).
chmod 755 /home/docker

if [ $# -eq 0 ]; then
  set -- bash
fi

if command -v setpriv >/dev/null 2>&1; then
  exec setpriv --reuid="${TARGET_UID}" --regid="${TARGET_GID}" --init-groups -- "$@"
fi

if ! id -u "${TARGET_UID}" >/dev/null 2>&1; then
  groupadd -g "${TARGET_GID}" hostmap 2>/dev/null || true
  useradd -u "${TARGET_UID}" -g "${TARGET_GID}" -M -s /bin/bash hostmap 2>/dev/null || true
fi

exec runuser -u "${TARGET_UID}" -- "$@"
