#!/usr/bin/env bash
# Clone LeRobot at the pinned upstream commit, copy this repo's files over it and install it.
#
#   bash setup.sh                 # -> ./lerobot, then `pip install -e`
#   bash setup.sh /path/to/dir    # custom checkout location
#   NO_INSTALL=1 bash setup.sh    # only clone + copy, skip pip
set -euo pipefail

LEROBOT_COMMIT=4aaff99be4a1d81568c08c8f0296b41b40c99ec4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$HERE/lerobot}"

if [ ! -d "$TARGET/.git" ]; then
    # Skip git-lfs test artifacts, they are not needed to run anything here
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git "$TARGET"
fi

if [ -n "$(git -C "$TARGET" status --porcelain --untracked-files=no)" ]; then
    echo "error: $TARGET has uncommitted changes, refusing to overwrite them" >&2
    exit 1
fi
git -C "$TARGET" fetch --quiet origin "$LEROBOT_COMMIT" || true
git -C "$TARGET" checkout --quiet -B Learn_SO101 "$LEROBOT_COMMIT"

cp -r "$HERE/overlay/." "$TARGET/"
echo "Copied $(find "$HERE/overlay" -type f | wc -l) files into $TARGET (lerobot @ ${LEROBOT_COMMIT:0:8})"

if [ -z "${NO_INSTALL:-}" ]; then
    pip install -e "$TARGET[feetech,hilserl,smolvla]"
    pip install mujoco placo wandb imageio
fi

echo "Done. Run the commands in README.md from: $TARGET"
