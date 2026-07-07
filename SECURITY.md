# Security Policy

## Supported versions

Only the latest release on the main branch receives security fixes.
LTS support windows will be defined once the v0.1 stable release ships.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately via:

- **Email:** security@thrindex.com  *(replace with actual address before first public release)*
- **Subject line:** `[SECURITY] <one-line description>`

You will receive an acknowledgement within **72 hours** and a status update
within **14 days**.  Critical vulnerabilities affecting shipped artifacts are
eligible for a coordinated disclosure window of up to **90 days** before
public disclosure; the default is **90 days**.

We commit to a public post-mortem for any finding that affected a shipped
artifact.

## Scope

| In scope | Out of scope |
|---|---|
| `thrindex-numerics` and all open crates | Customer-proprietary configurations |
| The `.thx` artifact format and signing chain | Third-party vendor SDKs |
| The public conformance suite | Issues in end-user models |

## Our security model

See Playbook §25 for the full threat model.  Key commitments:

- Signing keys held on offline hardware tokens; key ceremonies documented.
- Reproducible builds: every tagged release can be rebuilt bit-for-bit.
- `cargo audit` and `cargo deny` gate every CI merge.
- SBOM published with every release.
- The SDK phones home nothing by default; telemetry is opt-in with a
  documented, public schema.
