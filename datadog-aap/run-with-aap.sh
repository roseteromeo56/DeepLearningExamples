#!/usr/bin/env sh
set -eu

usage() {
    cat <<'EOF'
Usage: DD_SERVICE=<service> ./datadog-aap/run-with-aap.sh <command> [args...]

Runs a Python web/API process with Datadog App and API Protection enabled.
Install datadog-aap/requirements.txt in the application's environment first.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 64
fi

if [ -z "${DD_SERVICE:-}" ]; then
    echo "run-with-aap: DD_SERVICE must identify the protected API" >&2
    exit 64
fi

if ! command -v ddtrace-run >/dev/null 2>&1; then
    echo "run-with-aap: ddtrace-run was not found; install datadog-aap/requirements.txt" >&2
    exit 69
fi

# AppSec must be enabled before Python imports ddtrace or the web framework.
: "${DD_APPSEC_ENABLED:=true}"
: "${DD_API_SECURITY_ENABLED:=true}"
: "${DD_APPSEC_AUTO_USER_INSTRUMENTATION_MODE:=safe}"
: "${DD_REMOTE_CONFIGURATION_ENABLED:=true}"

export DD_APPSEC_ENABLED
export DD_API_SECURITY_ENABLED
export DD_APPSEC_AUTO_USER_INSTRUMENTATION_MODE
export DD_REMOTE_CONFIGURATION_ENABLED

exec ddtrace-run "$@"
