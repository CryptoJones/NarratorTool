#!/usr/bin/env bash
#
# NarratorTool installer.
#
# Creates a virtualenv against a supported Python, installs the package and its extras
# into it, checks for the system binaries the pipeline shells out to, and prints the
# command to narrate a file.
#
# The default installs Chatterbox, the default TTS backend, plus every parser. Kokoro
# is NOT installed alongside it: the two engines pin different torch versions, and
# resolving them together downgrades one. Pick the engine, not both.
#
#   ./INSTALL.sh                        # every parser + Chatterbox (the default engine)
#   ./INSTALL.sh --extras all           # every parser + Kokoro (the light engine)
#   ./INSTALL.sh --extras chatterbox    # TTS + plain text only
#   ./INSTALL.sh --extras pdf,epub      # parsers only, no torch download
#   ./INSTALL.sh --dev              # add pytest + ruff
#   ./INSTALL.sh --link             # also symlink `narrate` into ~/.local/bin
#   ./INSTALL.sh --recreate         # rebuild .venv from scratch
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
LINK_DIR="${HOME}/.local/bin"

EXTRAS="chatterbox,pdf,epub,docx,html"
WITH_DEV=0
DO_LINK=0
RECREATE=0

# Kokoro publishes no wheels for 3.13+, and the package floor is 3.10. Chatterbox is
# happy on 3.11 and 3.12, so this ordering suits both engines.
PY_CANDIDATES=(python3.12 python3.11 python3.10 python3)

# ---------------------------------------------------------------- output helpers
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    RED=$'\033[31m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; RESET=""
fi
step() { printf '%s==>%s %s%s%s\n' "$GREEN" "$RESET" "$BOLD" "$*" "$RESET"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '%s warn%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%serror%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
    sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# ---------------------------------------------------------------- arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --extras)   EXTRAS="${2:-}"; [ -n "$EXTRAS" ] || die "--extras needs a value"; shift 2 ;;
        --extras=*) EXTRAS="${1#*=}"; shift ;;
        --dev)      WITH_DEV=1; shift ;;
        --link)     DO_LINK=1; shift ;;
        --recreate) RECREATE=1; shift ;;
        -h|--help)  usage ;;
        *)          die "unknown option: $1 (try --help)" ;;
    esac
done

cd "$REPO_DIR"

# ---------------------------------------------------------------- 1. interpreter
step "Looking for a supported Python (>=3.10, <3.13)"

python_ok() {
    "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)' 2>/dev/null
}

PYTHON=""
for candidate in "${PY_CANDIDATES[@]}"; do
    path="$(command -v "$candidate" 2>/dev/null || true)"
    [ -n "$path" ] || continue
    if python_ok "$path"; then
        PYTHON="$path"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    found="$(command -v python3 >/dev/null 2>&1 && python3 -V 2>&1 || echo 'none found')"
    printf '%s\n' "" >&2
    die "no Python between 3.10 and 3.12 on PATH (system python3: $found).
      Kokoro ships no wheels for 3.13+, so the TTS extra cannot install on a newer one.
      Install an older interpreter, e.g.:
          sudo apt install python3.12 python3.12-venv     # Debian/Ubuntu
          brew install python@3.12                        # macOS
      or let uv fetch one for you:
          uv python install 3.12 && ./INSTALL.sh"
fi
info "using $PYTHON ($("$PYTHON" -V 2>&1))"

# uv is dramatically faster at resolving torch; use it when it is already installed.
UV="$(command -v uv 2>/dev/null || true)"
[ -n "$UV" ] && info "uv detected — using it for venv creation and installs"

# ---------------------------------------------------------------- 2. virtualenv
if [ "$RECREATE" -eq 1 ] && [ -d "$VENV" ]; then
    step "Removing existing virtualenv"
    rm -rf "$VENV"
fi

if [ -x "$VENV/bin/python" ]; then
    step "Reusing existing virtualenv at .venv"
    if ! python_ok "$VENV/bin/python"; then
        die ".venv runs $("$VENV/bin/python" -V 2>&1), which Kokoro does not support.
      Rebuild it against a supported interpreter:
          ./INSTALL.sh --recreate"
    fi
    info "$("$VENV/bin/python" -V 2>&1)"
else
    step "Creating virtualenv at .venv"
    if [ -n "$UV" ]; then
        "$UV" venv --python "$PYTHON" "$VENV"
    else
        "$PYTHON" -m venv "$VENV" || die "python -m venv failed.
      On Debian/Ubuntu the venv module ships separately:
          sudo apt install $(basename "$PYTHON")-venv"
    fi
fi

VENV_PY="$VENV/bin/python"

# ---------------------------------------------------------------- 3. package
SPEC="."
if [ -n "$EXTRAS" ] && [ "$EXTRAS" != "none" ]; then
    SPEC=".[$EXTRAS]"
fi
if [ "$WITH_DEV" -eq 1 ]; then
    SPEC="${SPEC%]}"
    case "$SPEC" in
        .) SPEC=".[dev]" ;;
        *) SPEC="$SPEC,dev]" ;;
    esac
fi

step "Installing narratortool $SPEC"
case "$EXTRAS" in
    *all*|*kokoro*|*chatterbox*) info "${DIM}this pulls torch — expect a large download on a cold cache${RESET}" ;;
esac

if [ -n "$UV" ]; then
    VIRTUAL_ENV="$VENV" "$UV" pip install -e "$SPEC"
else
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
    "$VENV_PY" -m pip install -e "$SPEC"
fi

# ---------------------------------------------------------------- 4. system binaries
step "Checking system dependencies"

case "$(uname -s)" in
    Darwin) INSTALL_HINT="brew install ffmpeg espeak-ng" ;;
    *)
        if   command -v apt-get >/dev/null 2>&1; then INSTALL_HINT="sudo apt install ffmpeg espeak-ng"
        elif command -v dnf     >/dev/null 2>&1; then INSTALL_HINT="sudo dnf install ffmpeg espeak-ng"
        elif command -v pacman  >/dev/null 2>&1; then INSTALL_HINT="sudo pacman -S ffmpeg espeak-ng"
        elif command -v zypper  >/dev/null 2>&1; then INSTALL_HINT="sudo zypper install ffmpeg espeak-ng"
        else INSTALL_HINT="install ffmpeg and espeak-ng with your package manager"
        fi
        ;;
esac

MISSING=()
# ffmpeg does the MP3 encode; without it the run dies at the final step, after
# synthesis, which is the most expensive thing to have to repeat.
if command -v ffmpeg >/dev/null 2>&1; then info "ffmpeg     found"; else info "ffmpeg     ${RED}MISSING${RESET}"; MISSING+=(ffmpeg); fi
# espeak-ng is Kokoro's grapheme-to-phoneme fallback for out-of-dictionary words, and
# is required for every non-American Kokoro voice — including bf_emma. Chatterbox has
# no phonemizer and does not need it, so it is only reported as missing when Kokoro is
# actually being installed.
if command -v espeak-ng >/dev/null 2>&1; then
    info "espeak-ng  found"
else
    case "$EXTRAS" in
        *all*|*kokoro*) info "espeak-ng  ${RED}MISSING${RESET}"; MISSING+=(espeak-ng) ;;
        *) info "espeak-ng  ${DIM}not installed (only Kokoro needs it)${RESET}" ;;
    esac
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    warn "missing: ${MISSING[*]} — install them before narrating:"
    warn "    $INSTALL_HINT"
fi

# ---------------------------------------------------------------- 5. verify
step "Verifying the install"
[ -x "$VENV/bin/narrate" ] || die "the narrate entry point was not created in .venv/bin"
"$VENV/bin/narrate" --version

# ---------------------------------------------------------------- 6. optional symlink
LINKED=0
if [ "$DO_LINK" -eq 1 ]; then
    step "Linking narrate into $LINK_DIR"
    mkdir -p "$LINK_DIR"
    ln -sf "$VENV/bin/narrate" "$LINK_DIR/narrate"
    LINKED=1
    case ":$PATH:" in
        *":$LINK_DIR:"*) info "$LINK_DIR is on your PATH" ;;
        *) warn "$LINK_DIR is not on your PATH — add it in your shell rc:"
           warn "    export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
    esac
fi

# ---------------------------------------------------------------- 7. how to run
REL_VENV="${VENV#"$PWD"/}"

cat <<EOF

${GREEN}${BOLD}Done.${RESET} narratortool is installed in ${BOLD}${REL_VENV}${RESET}.

${BOLD}Narrate a file${RESET} — no activation needed, just call the venv's copy:

    ${CYAN}${REL_VENV}/bin/narrate mydocument.pdf${RESET}

That writes ${BOLD}mydocument.mp3${RESET} next to the input, in the house voice
(bf_emma, UK female, speed 0.88). Inputs: .txt .md .pdf .epub .docx .html

${BOLD}Or activate the venv first${RESET}, and the short name works for the whole shell session:

    ${CYAN}source ${REL_VENV}/bin/activate${RESET}
    ${CYAN}narrate mydocument.pdf${RESET}
EOF

if [ "$LINKED" -eq 1 ]; then
cat <<EOF

${BOLD}You linked it${RESET}, so from any directory you can also just run:

    ${CYAN}narrate mydocument.pdf${RESET}
EOF
else
cat <<EOF

To type plain ${BOLD}narrate${RESET} from anywhere without activating, re-run with ${BOLD}--link${RESET}
(symlinks it into ${LINK_DIR}), or do it by hand:

    ${CYAN}ln -sf "$VENV/bin/narrate" "$LINK_DIR/narrate"${RESET}
EOF
fi

cat <<EOF

${BOLD}Useful flags${RESET}

    --dry-run              parse and chunk only, report stats, synthesize nothing
                           ${DIM}— run this first on a long book to see the chunk count${RESET}
    -o out/book.mp3        choose the output path
    --voice bm_george      a different voice; ${CYAN}--list-voices${RESET} shows them all
    --speed 1.0            faster or slower than the 0.88 default
    --announce-chapters    speak each chapter title before its body

${BOLD}Try it now${RESET}

    ${CYAN}${REL_VENV}/bin/narrate --list-voices${RESET}
    ${CYAN}${REL_VENV}/bin/narrate README.md --dry-run${RESET}
EOF

if [ ${#MISSING[@]} -gt 0 ]; then
cat <<EOF

${YELLOW}${BOLD}Before you narrate for real${RESET}, install the missing system package(s):

    ${CYAN}${INSTALL_HINT}${RESET}
EOF
fi
