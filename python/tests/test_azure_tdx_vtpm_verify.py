"""AzureTdxVtpmVerifier: the Azure Intel TDX measured-launch bundle, fail-closed.

Hermetic. A synthetic HCL blob (TD report TYPE byte 0x81, runtime-data JSON
carrying an RSA ``HCLAkPub``) is paired with a synthetic Intel-shaped DCAP quote
from ``tests.test_tdx.build_quote`` whose REPORT_DATA binds that runtime JSON,
plus an AK-signed SHA-256 PCR 23 quote built the way ``test_azure_vtpm_verify``
builds the SEV-SNP one. Every refusal in the verifier's ordered check list gets
its own mutation, and one case drives the REAL committed Azure DCAP quote
through the same path so the Intel chain step is exercised against genuine
silicon output.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import pathlib

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

from tests.test_tdx import NOW as TDX_NOW
from tests.test_tdx import build_quote
from wcm import AzureTdxVtpmVerifier, TrustStore, parse_tdx_quote
from wcm import azure_vtpm
from wcm.azure_vtpm import TDX_VTPM_BUNDLE_KIND, expected_pcr23_digest
from wcm.cli import INTEL_SGX_ROOT_CA_SHA256

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
AZURE_TDX = FIXTURES / "tdx_quote_azure.json"

NONCE = "ab" * 32
BINDING = b"\xcd" * 32
MEASUREMENT = "sha256:" + "42" * 32

# HCL layout, shared by the Azure SEV-SNP and TDX paravisors.
_REPORT_OFFSET = 32
_REPORT_SLOT = 1184
_RUNTIME_OFFSET = _REPORT_OFFSET + _REPORT_SLOT


def _tpm2b(value: bytes) -> bytes:
    return len(value).to_bytes(2, "big") + value


def _runtime_json(modulus: int) -> bytes:
    n = base64.urlsafe_b64encode(modulus.to_bytes(256, "big")).rstrip(b"=").decode()
    return json.dumps(
        {
            "keys": [{"kid": "HCLAkPub", "kty": "RSA", "n": n, "e": "AQAB"}],
            "vm-configuration": {},
            "user-data": "",
        },
        separators=(",", ":"),
    ).encode()


def _hcl(runtime_json: bytes, *, magic: bytes = b"HCLA", report_type: int = 0x81) -> bytes:
    report = bytearray(_REPORT_SLOT)
    report[0] = report_type  # REPORTMACSTRUCT TYPE: 0x81 is a TDX TD report
    runtime = bytes(16) + len(runtime_json).to_bytes(4, "little") + runtime_json
    return magic + bytes(28) + bytes(report) + runtime


def _tpm_quote(ak_key, *, nonce: str, no_pcr23: bool, pcr_digest: bytes) -> tuple[bytes, bytes]:
    extra = hashlib.sha256(bytes.fromhex(nonce) + BINDING).digest()
    selection = b"\x00\x0b\x03" + (b"\x01\x00\x00" if no_pcr23 else b"\x00\x00\x80")
    quote = (
        b"\xffTCG\x80\x18"
        + _tpm2b(b"signer")
        + _tpm2b(extra)
        + bytes(25)
        + (1).to_bytes(4, "big")
        + selection
        + _tpm2b(pcr_digest)
    )
    signature_blob = b"\x00\x14\x00\x0b" + _tpm2b(
        ak_key.sign(quote, padding.PKCS1v15(), hashes.SHA256())
    )
    return quote, signature_blob


def _pem(public_key) -> str:
    return public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()


def _bundle(
    *,
    kind: object = TDX_VTPM_BUNDLE_KIND,
    wrong_binding: bool = False,
    wrong_ak: bool = False,
    no_pcr23: bool = False,
    pcr_digest: bytes | None = None,
    non_rsa_ak: bool = False,
    magic: bytes = b"HCLA",
    report_type: int = 0x81,
    runtime_json_override: bytes | None = None,
    truncate_runtime_to: int | None = None,
    tamper_dcap: bool = False,
    bad_tpm_signature: bool = False,
    drop_member: str | None = None,
    trusted: bool = True,
) -> tuple[str, TrustStore]:
    """Assemble a wcm-azure-tdx-vtpm/v1 bundle and the trust store that fits it."""
    ak_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    runtime_key = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048) if wrong_ak else ak_key
    )
    runtime_json = (
        _runtime_json(runtime_key.public_key().public_numbers().n)
        if runtime_json_override is None
        else runtime_json_override
    )

    # REPORT_DATA[:32] = sha256(runtime JSON), tail zero: the Azure paravisor
    # binding, reproduced inside a synthetic Intel-signed DCAP quote.
    dcap, root = build_quote(nonce_hex=runtime_json.hex())
    if tamper_dcap:
        bad = bytearray(dcap)
        bad[48 + 136] ^= 0xFF  # an MRTD byte inside the attestation-key-signed body
        dcap = bytes(bad)

    hcl = _hcl(runtime_json, magic=magic, report_type=report_type)
    if truncate_runtime_to is not None:
        hcl = hcl[: _RUNTIME_OFFSET + truncate_runtime_to]

    quote, signature_blob = _tpm_quote(
        ak_key,
        nonce="ef" * 32 if wrong_binding else NONCE,
        no_pcr23=no_pcr23,
        pcr_digest=expected_pcr23_digest(MEASUREMENT) if pcr_digest is None else pcr_digest,
    )
    if bad_tpm_signature:
        flipped = bytearray(signature_blob)
        flipped[-1] ^= 0xFF
        signature_blob = bytes(flipped)

    ak_pem = (
        _pem(ec.generate_private_key(ec.SECP256R1()).public_key())
        if non_rsa_ak
        else _pem(ak_key.public_key())
    )
    doc: dict[str, object] = {
        "kind": kind,
        "tdx_quote_b64": base64.b64encode(dcap).decode(),
        "hcl_b64": base64.b64encode(hcl).decode(),
        "ak_pem": ak_pem,
        "tpm_quote_b64": base64.b64encode(quote).decode(),
        "tpm_signature_b64": base64.b64encode(signature_blob).decode(),
    }
    if drop_member is not None:
        del doc[drop_member]

    trust = TrustStore()
    if trusted:
        trust.add_root(root)
    return base64.b64encode(json.dumps(doc).encode()).decode(), trust


def _verify(bundle: str, trust: TrustStore, **overrides):
    kwargs = {
        "expected_nonce": NONCE,
        "channel_binding": BINDING,
        "expected_workload_measurement": MEASUREMENT,
        "now": TDX_NOW,
    }
    kwargs.update(overrides)
    return AzureTdxVtpmVerifier(trust).verify(bundle, **kwargs)


# -- happy path ---------------------------------------------------------------


def test_azure_tdx_vtpm_full_chain_verifies():
    bundle, trust = _bundle()
    result = _verify(bundle, trust)
    assert result.verified, result.reason
    assert "Intel SGX PCK" in (result.leaf_subject or "")


# -- bundle shape -------------------------------------------------------------


def test_azure_tdx_vtpm_rejects_wrong_kind():
    bundle, trust = _bundle(kind="wcm-azure-snp-vtpm/v1")
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "unexpected Azure TDX vTPM bundle kind: 'wcm-azure-snp-vtpm/v1'"


def test_azure_tdx_vtpm_rejects_malformed_bundle():
    bundle, trust = _bundle(drop_member="tdx_quote_b64")
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "invalid Azure vTPM evidence: 'tdx_quote_b64'"


def test_azure_tdx_vtpm_rejects_non_rsa_ak():
    bundle, trust = _bundle(non_rsa_ak=True)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "HCL AK is not RSA"


# -- HCL shape ----------------------------------------------------------------


def test_azure_tdx_vtpm_rejects_non_hcl():
    bundle, trust = _bundle(magic=b"NOPE")
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "not an Azure HCL report"


def test_azure_tdx_vtpm_rejects_snp_report_in_tdx_bundle():
    # TYPE 0x00 is not a TD report: an Azure SEV-SNP CVM shares this NV index.
    bundle, trust = _bundle(report_type=0x00)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "HCL does not wrap a TDX TD report"


# -- Intel DCAP chain ---------------------------------------------------------


def test_azure_tdx_vtpm_rejects_untrusted_intel_root():
    bundle, trust = _bundle(trusted=False)
    result = _verify(bundle, trust)
    assert not result.verified and "trusted root" in (result.reason or "")


def test_azure_tdx_vtpm_rejects_tampered_dcap_quote():
    bundle, trust = _bundle(tamper_dcap=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "quote signature does not verify under the attestation key"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_azure_tdx_vtpm_rejects_report_data_unbound_from_runtime_json():
    """A GENUINE Azure DCAP quote wrapped around someone else's runtime JSON.

    The quote itself verifies to Intel's real root, so reaching exactly this
    refusal proves the chain step ran and passed first: the paravisor binding is
    what fails, which is the substitution an attacker would attempt.
    """
    if not AZURE_TDX.exists():
        pytest.skip("no captured Azure TDX quote committed")
    fixture = json.loads(AZURE_TDX.read_text(encoding="utf-8"))
    assert fixture["intel_sgx_root_ca_sha256"] == INTEL_SGX_ROOT_CA_SHA256
    real = base64.b64decode(fixture["quote_b64"])

    parsed = parse_tdx_quote(real)
    chain = [parsed.pck_leaf, *parsed.pck_intermediates]
    root = next(c for c in chain if c.subject == c.issuer)
    assert (
        hashlib.sha256(root.public_bytes(serialization.Encoding.DER)).hexdigest()
        == INTEL_SGX_ROOT_CA_SHA256
    )
    trust = TrustStore()
    trust.add_root(root)
    # Deterministic clock inside the real leaf's validity window.
    leaf = parsed.pck_leaf
    now = leaf.not_valid_before_utc + (leaf.not_valid_after_utc - leaf.not_valid_before_utc) / 2

    synthetic, _ = _bundle()
    doc = json.loads(base64.b64decode(synthetic))
    doc["tdx_quote_b64"] = fixture["quote_b64"]
    bundle = base64.b64encode(json.dumps(doc).encode()).decode()

    result = _verify(bundle, trust, now=now)
    assert not result.verified
    assert result.reason == "TDX REPORT_DATA does not bind HCL runtime JSON"


# -- HCL runtime data and the AK link -----------------------------------------


def test_azure_tdx_vtpm_rejects_truncated_runtime_data():
    bundle, trust = _bundle(truncate_runtime_to=10)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "HCL runtime data is truncated"


def test_azure_tdx_vtpm_rejects_truncated_runtime_json():
    bundle, trust = _bundle(truncate_runtime_to=40)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "HCL runtime JSON is truncated"


def test_azure_tdx_vtpm_rejects_unlinked_ak():
    bundle, trust = _bundle(wrong_ak=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM quote key does not match HCL-authenticated AK"


# -- the vTPM PCR 23 quote ----------------------------------------------------


def test_azure_tdx_vtpm_rejects_wrong_nonce_binding():
    bundle, trust = _bundle(wrong_binding=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM qualifying data does not bind nonce and transport key"


def test_azure_tdx_vtpm_requires_pcr23():
    bundle, trust = _bundle(no_pcr23=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM quote does not select SHA-256 PCR 23"


def test_azure_tdx_vtpm_rejects_changed_pcr_state():
    bundle, trust = _bundle(pcr_digest=b"\xff" * 32)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM PCR digest does not match the approved workload measurement"


def test_azure_tdx_vtpm_rejects_wrong_policy_measurement():
    bundle, trust = _bundle()
    result = _verify(bundle, trust, expected_workload_measurement="sha256:" + "43" * 32)
    assert not result.verified
    assert result.reason == "TPM PCR digest does not match the approved workload measurement"


def test_azure_tdx_vtpm_rejects_malformed_pcr_digest():
    bundle, trust = _bundle(pcr_digest=b"short")
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == (
        "TPM quote verification failed: TPM PCR digest is not 32-byte SHA-256"
    )


def test_azure_tdx_vtpm_requires_policy_measurement():
    bundle, trust = _bundle()
    result = _verify(bundle, trust, expected_workload_measurement=None)
    assert not result.verified
    assert result.reason == "expected workload measurement is required by release policy"


def test_azure_tdx_vtpm_rejects_bad_tpm_signature():
    bundle, trust = _bundle(bad_tpm_signature=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert (result.reason or "").startswith("TPM quote verification failed:")


def test_azure_tdx_vtpm_rejects_non_object_bundle():
    bundle = base64.b64encode(json.dumps(["not", "an", "object"]).encode()).decode()
    result = _verify(bundle, TrustStore())
    assert not result.verified
    assert result.reason == "invalid Azure vTPM evidence: bundle is not a JSON object"


def test_azure_tdx_vtpm_rejects_unparseable_dcap_quote():
    bundle, trust = _bundle()
    doc = json.loads(base64.b64decode(bundle))
    doc["tdx_quote_b64"] = base64.b64encode(bytes(100)).decode()
    result = _verify(base64.b64encode(json.dumps(doc).encode()).decode(), trust)
    assert not result.verified and result.reason


def test_azure_tdx_vtpm_rejects_invalid_hclakpub():
    # Runtime JSON the TD report genuinely binds, but with no HCLAkPub to link
    # the quoting key to: the AK is then unauthenticated and the bundle fails.
    bundle, trust = _bundle(
        runtime_json_override=json.dumps({"keys": [], "user-data": ""}).encode()
    )
    result = _verify(bundle, trust)
    assert not result.verified
    assert (result.reason or "").startswith("invalid HCLAkPub:")


def test_azure_tdx_vtpm_rejects_nonzero_report_data_tail(monkeypatch):
    """The 32 REPORT_DATA bytes past the runtime-data hash must be zero.

    Azure fills only the first half. Reaching this branch needs a REPORT_DATA the
    attestation key still signs, so the parser the verifier calls is stubbed to
    dirty the tail while verify_tdx_quote still sees the genuine bytes.
    """
    real_parse = azure_vtpm.parse_tdx_quote

    def dirty_tail(quote: bytes):
        parsed = real_parse(quote)
        report = dataclasses.replace(
            parsed.report, report_data=parsed.report.report_data[:32] + b"" * 32
        )
        return dataclasses.replace(parsed, report=report)

    monkeypatch.setattr(azure_vtpm, "parse_tdx_quote", dirty_tail)
    bundle, trust = _bundle()
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "unexpected nonzero TDX REPORT_DATA tail"


# -- public unwrap helper ------------------------------------------------------


def test_unwrap_azure_tdx_vtpm_bundle_round_trips_every_member():
    from wcm import AzureTdxVtpmBundle, unwrap_azure_tdx_vtpm_bundle

    bundle_b64, _ = _bundle()
    doc = json.loads(base64.b64decode(bundle_b64))
    unwrapped = unwrap_azure_tdx_vtpm_bundle(bundle_b64)
    assert isinstance(unwrapped, AzureTdxVtpmBundle)
    assert unwrapped.dcap_quote == base64.b64decode(doc["tdx_quote_b64"])
    assert unwrapped.hcl == base64.b64decode(doc["hcl_b64"])
    assert unwrapped.ak_pem == doc["ak_pem"]
    assert unwrapped.tpm_quote == base64.b64decode(doc["tpm_quote_b64"])
    assert unwrapped.tpm_signature == base64.b64decode(doc["tpm_signature_b64"])


@pytest.mark.parametrize(
    "quote_b64, expected",
    [
        ("not-base64!", "invalid Azure vTPM evidence"),
        (base64.b64encode(b"[1, 2]").decode(), "bundle is not a JSON object"),
        (base64.b64encode(b'{"kind": "wcm-azure-snp-vtpm/v1"}').decode(), "unexpected Azure TDX vTPM bundle kind"),
        (base64.b64encode(b'{"kind": "wcm-azure-tdx-vtpm/v1", "hcl_b64": ""}').decode(), "invalid Azure vTPM evidence"),
    ],
)
def test_unwrap_azure_tdx_vtpm_bundle_refuses_malformed_input(quote_b64, expected):
    from wcm import unwrap_azure_tdx_vtpm_bundle

    with pytest.raises(ValueError, match=expected):
        unwrap_azure_tdx_vtpm_bundle(quote_b64)
