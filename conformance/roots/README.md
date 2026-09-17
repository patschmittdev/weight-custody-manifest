# Root store

Roots a conformance runner chains vendor captures to. A `vendor` vector names
its root by the SHA-256 of the root's DER and the runner resolves it here; it
never carries the root itself, and nothing in the suite fetches one.

## What is in this directory

| File | Subject | DER SHA-256 | Why it is here |
| --- | --- | --- | --- |
| `out-of-chain-root.pem` | `CN=wcm-conformance-out-of-chain-root, O=WCM conformance suite` | `e2993cf147a1b4198a2cbe4d9e7517082773b0cbdcb5186099e7aafe3acbbdc2` | The untrusted anchor in the refusal matrix. Shipped so every runner fails that case for the same reason. |
| `intel-sgx-root-ca.pem` | `C=US, ST=CA, L=Santa Clara, O=Intel Corporation, CN=Intel SGX Root CA` | `44a0196b2b99f889b8e149e95b807a350e7424964399e885a7cbb8ccfab674d3` | The anchor for the committed TDX capture. Intel publishes it, it is the same digest `wcm.cli` already pins as `INTEL_SGX_ROOT_CA_SHA256`, and a vendor vector whose root nobody staged fails the reference self-test for a reason about staging rather than about the capture. |

Staging Intel's root here rather than leaving it to `WCM_CONFORMANCE_ROOTS` is a
deliberate exception to the rule below, and it is only sound because the digest
is pinned out of band: `wcm.cli.INTEL_SGX_ROOT_CA_SHA256` carries the same value
independently of any vector, so the capture is not anchoring itself. Read this
row as one trust decision, not as permission to add the next root by copying it.

The out-of-chain root is a test certificate. It signs nothing, it anchors
nothing, and it protects nothing. It exists so that the `out-of-chain-root`
refusal case is the same experiment everywhere.

It is shipped rather than generated per runner because both of the obvious ways
to invent it quietly test nothing:

- an anchor taken from the capture's own chain verifies correctly, because it
  is a legitimate anchor;
- a VCEK offered as a root is rejected on certificate policy for a non-positive
  serial *before any chain logic runs*, so it fails for the wrong reason.

Both produce a refusal, and neither is the refusal the case is about.

## Staging vendor roots

Vendor roots are not all redistributable, so the suite names them and the runner
stages them. Point `WCM_CONFORMANCE_ROOTS` at one or more directories of PEM
files:

```sh
export WCM_CONFORMANCE_ROOTS=/etc/wcm/roots:$HOME/.wcm/roots
wcm conformance
```

Every `*.pem` in those directories is indexed by the SHA-256 of its DER. A file
may hold more than one certificate. Indexing is by digest rather than by subject
because a subject is a claim inside the certificate and a digest is the
certificate: two roots can share a common name, and they cannot share a digest.

A vector whose root is not staged fails with `root not staged`, naming the
digest. That is deliberate and it is not an inconvenience to route around:

- a suite that fetched a root mid-run would fail differently on an aeroplane
  than in CI, and the failure would be about the network rather than about the
  implementation;
- a vector carrying its own anchor proves only that the vector is internally
  consistent, which the synthetic PKI in `vectors/gate/` already proves;
- staging a vendor root is a step every real implementer performs anyway.

Each vector records where its root comes from in `chain.root_source`. That field
is documentation for a person. Nothing reads it as a URL.

## Adding a root here

Only for roots this project may redistribute, and only with the table above
updated: file, subject, digest, and why. A root in this directory is trusted by
every runner that uses the suite, so an unexplained one is a supply chain
question nobody can answer later.
