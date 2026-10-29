# Source from bash (or via .envrc): builds the Guix profile, then syncs and activates .venv.
root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
venv="$root/.venv"

# GC-rooted profile at a stable path, so the venv's base python survives guix gc. guix shell --root will not
# replace an existing link ("File exists" after a manifest change), so drop the old one first.
rm -f "$root/.guix-profile"
guix shell -m "$root/manifest.scm" --root="$root/.guix-profile" -- true || return 1
GUIX_PROFILE="$root/.guix-profile"
. "$GUIX_PROFILE/etc/profile"

# Use the profile's python only; uv-downloaded pythons expect /lib/ld-linux and cannot run on Guix System.
export UV_PYTHON="$GUIX_PROFILE/bin/python3" UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT="$venv"
"$GUIX_PROFILE/bin/uv" sync --frozen --quiet --project "$root" || return 1

# manylinux wheels need libgcc_s. Preload it inside the venv's python only; a global LD_LIBRARY_PATH
# would hand the profile's glibc to host binaries.
echo "import ctypes; ctypes.CDLL('$GUIX_PROFILE/lib/libgcc_s.so.1', ctypes.RTLD_GLOBAL)" \
  > "$venv/lib/python3.12/site-packages/guix-libs.pth"

. "$venv/bin/activate"
