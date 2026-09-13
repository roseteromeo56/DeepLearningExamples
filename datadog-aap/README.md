# Datadog App and API Protection

This directory provides an opt-in launcher for Python HTTP services in this
repository. It enables Datadog Application Security Management (AppSec), API
Security endpoint discovery, automated user-event tracking, and remote
configuration. Training and batch workloads remain unchanged.

## Prerequisites

- A Datadog Agent reachable from the application with APM enabled. Containerized
  Agents must listen on a non-loopback interface (for example,
  `DD_APM_NON_LOCAL_TRAFFIC=true`).
- App and API Protection enabled for the service in Datadog.
- A supported Python web framework. The native Triton Inference Server process
  is not a Python web framework; protect a supported Python API/gateway in front
  of Triton instead.

Do not put a Datadog API key in the application container. Only the Agent needs
the key; the application sends traces to the Agent on port 8126.

## Run a protected API

Install the tracer in the same environment as the API:

```sh
python -m pip install -r datadog-aap/requirements.txt
```

Copy the example settings into your deployment's secret/configuration system,
adjust the service tags and Agent host, then launch the existing API command:

```sh
set -a
. datadog-aap/env.example
set +a
./datadog-aap/run-with-aap.sh python path/to/api.py
```

For Gunicorn, keep `ddtrace-run` outside the process manager by using the same
launcher:

```sh
./datadog-aap/run-with-aap.sh gunicorn --bind 0.0.0.0:8080 package.app:app
```

The launcher defaults the four protection settings to secure values, but
respects values explicitly supplied by the deployment. `DD_SERVICE` is required
to prevent unrelated examples from being reported as one service. Set
`DD_ENV` and `DD_VERSION` as well so findings can be tied to a deployment.

## Verify

1. Send normal traffic to every API route after deployment.
2. Confirm the service and endpoint inventory appear in Datadog API Security.
3. In Datadog App and API Protection, use the built-in harmless test request to
   verify detection before enabling blocking rules.
4. Keep remote configuration enabled so blocking rules and protection updates
   can be applied without rebuilding the image.

Start in monitoring mode and review signals for false positives before enabling
blocking in production. Datadog's default obfuscation should remain enabled;
avoid custom request-header or body capture that could collect credentials or
model inputs.
