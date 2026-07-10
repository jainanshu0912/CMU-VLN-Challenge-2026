#!/bin/bash
set -e

# Map container user to host user so bind-mounted workspaces stay editable on the host.
TARGET_UID="${HOST_UID:-1000}"
TARGET_GID="${HOST_GID:-1000}"

if [ "$(id -u)" -ne 0 ]; then
  exec "$@"
fi

chmod 755 /home/docker

if [ -d /home/docker/ai_module ]; then
  chown -R "${TARGET_UID}:${TARGET_GID}" /home/docker/ai_module
fi

if [ $# -eq 0 ]; then
  set -- bash
fi

if command -v gosu >/dev/null 2>&1; then
  exec gosu "${TARGET_UID}:${TARGET_GID}" "$@"
fi

exec runuser -u "${TARGET_UID}" -g "${TARGET_GID}" -- "$@"
