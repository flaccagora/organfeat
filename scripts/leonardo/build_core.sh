#!/usr/bin/env bash
# Build core.sif on LEONARDO without root or fakeroot (neither is available to users):
# Ubuntu 24.04 sandbox + runtime libs open3d needs (unpacked from .debs with dpkg-deb -x) + uv +
# a uv-managed CPython 3.12 under /opt/python, then squashed to $OFEAT_CORE_SIF.
# The final mksquashfs can exceed the login nodes' 10-min CPU cap and leave a truncated SIF (a few KB):
# then run it on the serial partition: `sbatch -A IscrC_FLAC -p lrd_all_serial -c 4 --wrap scripts/leonardo/build_core.sh`.
# The Python environment itself is NOT in the image: `scripts/leonardo/run.sh uv sync --frozen`
# builds .venv on the shared filesystem from uv.lock (ADR-0011).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/scripts/leonardo/env.sh"
W="$(mktemp -d -p "$SINGULARITY_TMPDIR")"; trap 'chmod -R u+w "$W"; rm -rf "$W"' EXIT
cd "$W"
singularity build --sandbox sb docker://ubuntu:24.04
mkdir -p apt/lists/partial apt/cache/archives/partial debs \
  sb/leonardo sb/leonardo_scratch sb/leonardo_work sb/scratch_local
touch sb/etc/localtime
PK="libegl1 libgl1 libx11-6 libusb-1.0-0 libgomp1 libglib2.0-0t64 ca-certificates git curl unzip gcc g++ libc6-dev make ninja-build"  # toolchain: nvdiffrast build (CUDA from the host cuda/12.6 module)
singularity exec -B "$W" sb bash -c "
  O='-o Dir::State::Lists=$W/apt/lists -o Dir::Cache=$W/apt/cache -o Debug::NoLocking=1 -o APT::Sandbox::User=$(id -un)'
  apt-get \$O update -qq
  ALL=\$(apt-cache \$O depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks \
         --no-replaces --no-enhances $PK | grep '^\w' | sort -u)
  cd debs && apt-get \$O download \$ALL"
for d in debs/*.deb; do singularity exec -B "$W" sb dpkg-deb -x "$d" sb; done
cp "$(command -v uv)" sb/usr/local/bin/
singularity exec --writable --no-home -B /leonardo_scratch sb bash -c '
  ldconfig
  cd /usr/share/ca-certificates && find . -name "*.crt" | sed "s|^\./||" > /etc/ca-certificates.conf
  update-ca-certificates
  UV_CACHE_DIR='"$UV_CACHE_DIR"' UV_PYTHON_INSTALL_DIR=/opt/python uv python install 3.12.11  # .venv links to this exact interpreter
  ln -sf /opt/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12 /usr/local/bin/python3  # download scripts call python3
  ln -sf python3 /usr/local/bin/python
  ln -sf g++ /usr/bin/c++; ln -sf gcc /usr/bin/cc  # dpkg-deb -x skips update-alternatives'
singularity build --force "$OFEAT_CORE_SIF" sb
echo "built $OFEAT_CORE_SIF"
