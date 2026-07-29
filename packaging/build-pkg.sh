#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT_MANIFEST="${REPO_ROOT}/Cargo.toml"
TUNER_MANIFEST="${REPO_ROOT}/plugins/mot-tuner/Cargo.toml"

manifest_version() {
    awk -F '"' '/^version = / { print $2; exit }' "$1"
}

ROOT_VERSION="$(manifest_version "${ROOT_MANIFEST}")"
TUNER_VERSION="$(manifest_version "${TUNER_MANIFEST}")"

if [[ -z "${ROOT_VERSION}" || -z "${TUNER_VERSION}" ]]; then
    echo "Cannot read package versions from Cargo manifests" >&2
    exit 1
fi
if [[ "${ROOT_VERSION}" != "${TUNER_VERSION}" ]]; then
    echo "Root and MOT TUNER versions must match for cargo-truce packaging" >&2
    echo "Root: ${ROOT_VERSION}; MOT TUNER: ${TUNER_VERSION}" >&2
    exit 1
fi
if [[ ! "${TUNER_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
    echo "MOT TUNER version is not valid SemVer: ${TUNER_VERSION}" >&2
    exit 1
fi

OUTPUT="${1:-${REPO_ROOT}/dist/MOT-TUNER-${TUNER_VERSION}-macOS-arm64.pkg}"
TRUCE_OUTPUT="${REPO_ROOT}/target/dist/mot-tuner-${TUNER_VERSION}-macos-user.pkg"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mot-tuner-pkg.XXXXXX")"

cleanup() {
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

(
    cd -- "${REPO_ROOT}"
    cargo metadata --locked --no-deps --format-version 1 >/dev/null
    cargo truce package \
        -p mot-tuner \
        --formats vst3 \
        --user \
        --host-only \
        --no-notarize
)

if [[ ! -f "${TRUCE_OUTPUT}" ]]; then
    echo "cargo-truce did not produce ${TRUCE_OUTPUT}" >&2
    exit 1
fi

EXPANDED="${WORK_DIR}/expanded"
/usr/sbin/pkgutil --expand-full "${TRUCE_OUTPUT}" "${EXPANDED}"

if ! grep -Fq 'enable_currentUserHome="true"' "${EXPANDED}/Distribution" ||
    ! grep -Fq 'enable_localSystem="false"' "${EXPANDED}/Distribution"
then
    echo "Installer is not restricted to the current-user domain" >&2
    exit 1
fi

PACKAGE_INFO_FILES="$(find "${EXPANDED}" -name PackageInfo -type f -print)"
PACKAGE_INFO_COUNT="$(find "${EXPANDED}" -name PackageInfo -type f -print | awk 'END { print NR }')"
if [[ "${PACKAGE_INFO_COUNT}" -ne 1 ]]; then
    echo "Expected exactly one component PackageInfo" >&2
    exit 1
fi
if ! grep -Fq 'identifier="com.plutandmot.mot-tuner.vst3"' "${PACKAGE_INFO_FILES}" ||
    ! grep -Fq "version=\"${TUNER_VERSION}\"" "${PACKAGE_INFO_FILES}" ||
    ! grep -Fq 'relocatable="false"' "${PACKAGE_INFO_FILES}" ||
    ! grep -Fq 'install-location="/Library/Audio/Plug-Ins/VST3/"' "${PACKAGE_INFO_FILES}"
then
    echo "Installer component metadata does not match the MOT TUNER contract" >&2
    exit 1
fi

PAYLOAD_BUNDLES="$(find "${EXPANDED}" -type d -name "MOT TUNER.vst3" -print)"
PAYLOAD_COUNT="$(find "${EXPANDED}" -type d -name "MOT TUNER.vst3" -print | awk 'END { print NR }')"
if [[ "${PAYLOAD_COUNT}" -ne 1 ]]; then
    echo "Expected exactly one MOT TUNER.vst3 payload" >&2
    exit 1
fi
if find "${EXPANDED}" -iname "MOT PLAYER*" -o -iname "MOT TRAINER*" -o -iname "MOT Guitar Plugin*" |
    grep -q .
then
    echo "Retired MOT plug-ins leaked into the installer payload" >&2
    exit 1
fi

PAYLOAD_BUNDLE="${PAYLOAD_BUNDLES}"
PAYLOAD_BINARY="${PAYLOAD_BUNDLE}/Contents/MacOS/MOT TUNER"

if [[ "$(/usr/bin/lipo -archs "${PAYLOAD_BINARY}")" != "arm64" ]]; then
    echo "Installer payload is not ARM64-only" >&2
    exit 1
fi
/usr/bin/codesign --verify --deep --strict --verbose=2 "${PAYLOAD_BUNDLE}"

mkdir -p "$(dirname -- "${OUTPUT}")"
TEMP_OUTPUT="${WORK_DIR}/$(basename -- "${OUTPUT}")"
/bin/cp "${TRUCE_OUTPUT}" "${TEMP_OUTPUT}"
/bin/mv -f "${TEMP_OUTPUT}" "${OUTPUT}"

if [[ "$(/usr/bin/stat -f '%z' "${OUTPUT}")" -le 51200 ]]; then
    echo "Installer package is unexpectedly small" >&2
    exit 1
fi

if /usr/sbin/pkgutil --check-signature "${OUTPUT}"; then
    echo "Installer package signature verified."
else
    echo "Installer package is unsigned because no Developer ID Installer identity is configured."
fi

echo
echo "Built ${OUTPUT}"
/usr/bin/shasum -a 256 "${OUTPUT}"
