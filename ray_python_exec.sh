#!/usr/bin/env bash
set -u

# Keep site-packages discoverable if we fall back to copied Python binaries.
add_site_packages() {
    local venv_path="$1"
    local site_pkg
    site_pkg=$(ls -d "${venv_path}"/lib/python3.*/site-packages 2>/dev/null | head -1 || true)
    if [ -n "${site_pkg}" ]; then
        export PYTHONPATH="${site_pkg}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
}

add_site_packages "/tmp/maor_venv_a"
add_site_packages "/tmp/maor_venv_b"
add_site_packages "${PWD}/venv"

if [ -x "/tmp/maor_py_a" ]; then
    exec /tmp/maor_py_a "$@"
fi
if [ -x "/tmp/maor_py_b" ]; then
    exec /tmp/maor_py_b "$@"
fi
if [ -x "/tmp/maor_venv_a/bin/python" ]; then
    exec /tmp/maor_venv_a/bin/python "$@"
fi
if [ -x "/tmp/maor_venv_b/bin/python" ]; then
    exec /tmp/maor_venv_b/bin/python "$@"
fi
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$@"
fi
exec python "$@"
