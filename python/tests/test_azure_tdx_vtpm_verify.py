"""AzureTdxVtpmVerifier against synthetic evidence, fail-closed.

The synthetic evidence is built the way the real evidence is related. A
1024-byte TDREPORT is assembled first (REPORTDATA = sha256(runtime JSON)); the
HCL wraps it beside the runtime JSON with consistent sizes and zero padding;
and the Intel-shaped DCAP quote from ``tests.test_tdx.build_quote`` signs a TD
report body *derived from that TDREPORT* through an offset table written out
here independently of the verifier's own. The AK signs a SHA-256 PCR 23 TPM
quote over ``sha256(nonce || binding)``.

Synthetic evidence is needed wherever a refusal depends on changing the runtime
JSON or the TD report: on a real capture both are bound by Intel's signature.
No real capture is committed (see CHANGELOG), so these bundles are the
public coverage.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

from tests.test_azure_vtpm_verify import _SHA256_DIGEST_INFO, _b64url, _rsa_spki_pem, _tpm2b
from tests.test_tdx import NOW as TDX_NOW
from tests.test_tdx import build_quote
from wcm import (
    AzureTdxVtpmBundle,
    AzureTdxVtpmVerifier,
    TrustStore,
    unwrap_azure_tdx_vtpm_bundle,
)
from wcm import azure_vtpm
from wcm.azure_vtpm import TDX_VTPM_BUNDLE_KIND, expected_pcr23_digest

NONCE = "ab" * 32
BINDING = b"\xcd" * 32
MEASUREMENT = "sha256:" + "42" * 32
MRTD = b"\x33" * 48
PARAVISOR = MRTD.hex()

# Azure HCL report layout: header, hardware-report slot, runtime data.
_REPORT_OFFSET = 32
_REPORT_SLOT = 1184
_RUNTIME_OFFSET = _REPORT_OFFSET + _REPORT_SLOT

# TDREPORT_STRUCT offsets of each DCAP v4 TD report body field:
# (name, body offset, length, TDREPORT offset).
TDREPORT_FIELDS = (
    ("TEE_TCB_SVN", 0, 16, 264),
    ("MRSEAM", 16, 48, 280),
    ("MRSIGNERSEAM", 64, 48, 328),
    ("SEAMATTRIBUTES", 112, 8, 376),
    ("TDATTRIBUTES", 120, 8, 512),
    ("XFAM", 128, 8, 520),
    ("MRTD", 136, 48, 528),
    ("MRCONFIGID", 184, 48, 576),
    ("MROWNER", 232, 48, 624),
    ("MROWNERCONFIG", 280, 48, 672),
    ("RTMR0-3", 328, 192, 720),
    ("REPORTDATA", 520, 64, 128),
)


def _runtime_json(modulus: int, jwk: dict | None = None) -> bytes:
    # Same shape as a real Azure HCL runtime JWK: kid, key_ops, kty, e, n.
    hcl_jwk = {
        "kid": "HCLAkPub",
        "key_ops": ["sign"],
        "kty": "RSA",
        "e": "AQAB",
        "n": _b64url(modulus, 256),
    }
    if jwk is not None:
        hcl_jwk = {k: v for k, v in {**hcl_jwk, **jwk}.items() if v is not None}
    return json.dumps(
        {"keys": [hcl_jwk], "vm-configuration": {}, "user-data": ""},
        separators=(",", ":"),
    ).encode()


def _tdreport(
    runtime_json: bytes,
    *,
    mrtd: bytes = MRTD,
    td_attributes: int = 0x10000000,
    report_data_tail: bytes = bytes(32),
) -> bytes:
    """A TDREPORT_STRUCT whose REPORTDATA binds ``runtime_json``."""
    report = bytearray(1024)
    report[0] = 0x81  # REPORTMACSTRUCT TYPE: TDX
    report[128:160] = hashlib.sha256(runtime_json).digest()  # REPORTDATA
    report[160:192] = report_data_tail  # zero on Azure
    report[264:280] = bytes([0x0F, 0x01, 0x0A]) + bytes(13)  # TEE_TCB_SVN
    report[280:328] = b"\x5e" * 48  # MRSEAM
    report[512:520] = td_attributes.to_bytes(8, "little")
    report[520:528] = bytes.fromhex("e718060000000000")  # XFAM
    report[528:576] = mrtd
    report[720:912] = bytes(range(192))  # RTMR0-3
    return bytes(report)


def dcap_body(tdreport: bytes) -> bytes:
    """The 584-byte DCAP v4 TD report body the TDREPORT corresponds to."""
    body = bytearray(584)
    for _, body_offset, length, tdreport_offset in TDREPORT_FIELDS:
        body[body_offset : body_offset + length] = tdreport[tdreport_offset : tdreport_offset + length]
    return bytes(body)


def _hcl(
    runtime_json: bytes,
    tdreport: bytes,
    *,
    magic: bytes = b"HCLA",
    padding_len: int = 134,
    report_size_delta: int = 0,
    data_size_delta: int = 0,
    report_type: int = 4,
    hash_type: int = 1,
    header_version: int = 2,
    request_type: int = 2,
    runtime_version: int = 1,
) -> bytes:
    """An HCL blob with the framing the real capture has."""
    data_size = 20 + len(runtime_json)
    report_size = _RUNTIME_OFFSET + data_size
    header = magic + struct.pack(
        "<7I", header_version, report_size + report_size_delta, request_type, 0, 0, 0, 0
    )
    slot = tdreport + bytes(_REPORT_SLOT - len(tdreport))
    runtime = struct.pack(
        "<5I",
        data_size + data_size_delta,
        runtime_version,
        report_type,
        hash_type,
        len(runtime_json),
    ) + runtime_json
    return header + slot + runtime + bytes(padding_len)


def _tpm_quote(
    nonce: str,
    *,
    pcr_digest: bytes,
    selections: bytes = b"\x00\x0b\x03\x00\x00\x80",
    count: int = 1,
) -> bytes:
    extra = hashlib.sha256(bytes.fromhex(nonce) + BINDING).digest()
    return (
        b"\xffTCG\x80\x18"
        + _tpm2b(b"signer")
        + _tpm2b(extra)
        + bytes(25)
        + count.to_bytes(4, "big")
        + selections
        + _tpm2b(pcr_digest)
    )


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
    selections: bytes | None = None,
    selection_count: int = 1,
    non_rsa_ak: bool = False,
    magic: bytes = b"HCLA",
    report_type_byte: int = 0x81,
    jwk: dict | None = None,
    runtime_json_override: bytes | None = None,
    ak_key: rsa.RSAPrivateKey | None = None,
    mrtd: bytes = MRTD,
    td_attributes: int = 0x10000000,
    report_data_tail: bytes = bytes(32),
    hcl_kwargs: dict | None = None,
    hcl_tdreport_change: tuple[int, int] | None = None,
    truncate_runtime_to: int | None = None,
    tamper_dcap: bool = False,
    bad_tpm_signature: bool = False,
    drop_member: str | None = None,
    trusted: bool = True,
) -> tuple[str, TrustStore]:
    """Assemble a wcm-azure-tdx-vtpm/v1 bundle and the trust store that fits it.

    The DCAP quote signs the body derived from the very TDREPORT the HCL
    carries; ``hcl_tdreport_change`` (offset, xor) then alters the HCL's copy
    only, after signing.
    """
    if ak_key is None:
        ak_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    runtime_key = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048) if wrong_ak else ak_key
    )
    runtime_json = (
        _runtime_json(runtime_key.public_key().public_numbers().n, jwk)
        if runtime_json_override is None
        else runtime_json_override
    )

    tdreport = _tdreport(
        runtime_json, mrtd=mrtd, td_attributes=td_attributes, report_data_tail=report_data_tail
    )
    dcap, root = build_quote(td_body=dcap_body(tdreport))
    if tamper_dcap:
        bad = bytearray(dcap)
        bad[48 + 136] ^= 0xFF  # an MRTD byte inside the attestation-key-signed body
        dcap = bytes(bad)

    hcl_tdreport = bytearray(tdreport)
    hcl_tdreport[0] = report_type_byte
    if hcl_tdreport_change is not None:
        offset, xor = hcl_tdreport_change
        hcl_tdreport[offset] ^= xor
    hcl = _hcl(runtime_json, bytes(hcl_tdreport), magic=magic, **(hcl_kwargs or {}))
    if truncate_runtime_to is not None:
        hcl = hcl[: _RUNTIME_OFFSET + truncate_runtime_to]

    if selections is None:
        selections = b"\x00\x0b\x03" + (b"\x01\x00\x00" if no_pcr23 else b"\x00\x00\x80")
    quote = _tpm_quote(
        "ef" * 32 if wrong_binding else NONCE,
        pcr_digest=expected_pcr23_digest(MEASUREMENT) if pcr_digest is None else pcr_digest,
        selections=selections,
        count=selection_count,
    )
    signature_blob = b"\x00\x14\x00\x0b" + _tpm2b(
        ak_key.sign(quote, padding.PKCS1v15(), hashes.SHA256())
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


def _verifier(trust: TrustStore, measurements=(PARAVISOR,)) -> AzureTdxVtpmVerifier:
    return AzureTdxVtpmVerifier(trust, paravisor_measurements=measurements)


def _verify(bundle: str, trust: TrustStore, *, measurements=(PARAVISOR,), **overrides):
    kwargs = {
        "expected_nonce": NONCE,
        "channel_binding": BINDING,
        "expected_workload_measurement": MEASUREMENT,
        "now": TDX_NOW,
    }
    kwargs.update(overrides)
    return _verifier(trust, measurements).verify(bundle, **kwargs)


def _redo(bundle: str, **changes) -> str:
    doc = json.loads(base64.b64decode(bundle))
    doc.update(changes)
    return base64.b64encode(json.dumps(doc).encode()).decode()


# -- happy path ---------------------------------------------------------------


def test_azure_tdx_vtpm_full_chain_verifies():
    bundle, trust = _bundle()
    result = _verify(bundle, trust)
    assert result.verified, result.reason
    assert "Intel SGX PCK" in (result.leaf_subject or "")


# -- bundle shape -------------------------------------------------------------


@pytest.mark.parametrize("member", ["kind", "tdx_quote_b64"])
def test_azure_tdx_vtpm_rejects_duplicate_bundle_member(member):
    bundle, trust = _bundle()
    assert _verify(bundle, trust).verified
    raw = base64.b64decode(bundle).decode()
    duplicate = json.dumps({member: "wrong"})[:-1] + "," + raw[1:]
    result = _verify(base64.b64encode(duplicate.encode()).decode(), trust)
    assert not result.verified
    assert result.reason == f"invalid Azure vTPM evidence: duplicate JSON object key {member!r}"


def test_azure_tdx_vtpm_rejects_wrong_kind():
    bundle, trust = _bundle(kind="wcm-azure-snp-vtpm/v1")
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "unexpected Azure TDX vTPM bundle kind: 'wcm-azure-snp-vtpm/v1'"


def test_azure_tdx_vtpm_rejects_missing_member():
    bundle, trust = _bundle(drop_member="tdx_quote_b64")
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "invalid Azure vTPM evidence: 'tdx_quote_b64'"


def test_azure_tdx_vtpm_rejects_non_rsa_ak():
    bundle, trust = _bundle(non_rsa_ak=True)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "HCL AK is not RSA"


def test_azure_tdx_vtpm_malformed_bundles_deny_instead_of_raising():
    bundle, trust = _bundle()
    cases = {
        "not base64": "not-base64!",
        "not a string": None,
        "not JSON": base64.b64encode(b"\xff\xfe{").decode(),
        "not an object": base64.b64encode(b"[1, 2]").decode(),
        "member not base64": _redo(bundle, hcl_b64="%%%"),
        "member not a string": _redo(bundle, tpm_quote_b64=7),
        "ak_pem not a string": _redo(bundle, ak_pem=["pem"]),
        "ak_pem not PEM": _redo(bundle, ak_pem="-----BEGIN PUBLIC KEY-----\nAA==\n-----END PUBLIC KEY-----\n"),
        "empty HCL": _redo(bundle, hcl_b64=""),
        "garbage DCAP quote": _redo(bundle, tdx_quote_b64=base64.b64encode(bytes(100)).decode()),
    }
    for label, quote_b64 in cases.items():
        result = _verify(quote_b64, trust)
        assert not result.verified, label
        assert result.reason, label


def test_azure_tdx_vtpm_dcap_check_errors_are_denials(monkeypatch):
    # The delegated DCAP check is inside the same denial boundary as the rest
    # of the bundle.
    def raises(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(azure_vtpm, "verify_tdx_quote", raises)
    bundle, trust = _bundle()
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "the DCAP quote could not be verified"


# -- HCL shape ----------------------------------------------------------------


def test_azure_tdx_vtpm_rejects_non_hcl():
    bundle, trust = _bundle(magic=b"NOPE")
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "not an Azure HCL report"


def test_azure_tdx_vtpm_rejects_snp_report_in_tdx_bundle():
    # TYPE 0x00 is not a TD report: an Azure SEV-SNP CVM shares this NV index.
    bundle, trust = _bundle(report_type_byte=0x00)
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


def test_azure_tdx_vtpm_requires_intel_td_quoting_enclave():
    # #163: the QE identity check in verify_tdx_quote runs on this path too.
    dcap, root = build_quote(nonce_hex="00", qe_prodid=1)
    bundle, _ = _bundle()
    bundle = _redo(bundle, tdx_quote_b64=base64.b64encode(dcap).decode())
    trust = TrustStore()
    trust.add_root(root)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "QE report product id is not Intel's TD Quoting Enclave"


# -- HCL runtime data and the AK link -----------------------------------------


def test_azure_tdx_vtpm_rejects_duplicate_runtime_kid():
    ak_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    runtime = _runtime_json(ak_key.public_key().public_numbers().n)
    bundle, trust = _bundle(ak_key=ak_key, runtime_json_override=runtime)
    assert _verify(bundle, trust).verified
    duplicate = runtime.replace(b'"kid":"HCLAkPub"', b'"kid":"wrong","kid":"HCLAkPub"', 1)
    # Both the TDREPORT and the DCAP signature bind the duplicate-bearing JSON.
    bundle, trust = _bundle(ak_key=ak_key, runtime_json_override=duplicate)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "invalid HCLAkPub: duplicate JSON object key 'kid'"


def test_azure_tdx_vtpm_rejects_truncated_runtime_data():
    bundle, trust = _bundle(truncate_runtime_to=10)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "HCL runtime data is truncated"


def test_azure_tdx_vtpm_rejects_truncated_runtime_json():
    bundle, trust = _bundle(truncate_runtime_to=40)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "HCL report is truncated"


def test_azure_tdx_vtpm_rejects_report_data_unbound_from_runtime_json():
    # The quote and the HCL's TDREPORT bind one runtime JSON; the HCL carries
    # another, naming a different AK.
    bundle, trust = _bundle()
    hcl = base64.b64decode(json.loads(base64.b64decode(bundle))["hcl_b64"])
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = _hcl(_runtime_json(other_key.public_key().public_numbers().n), hcl[32 : 32 + 1024])
    result = _verify(_redo(bundle, hcl_b64=base64.b64encode(other).decode()), trust)
    assert not result.verified
    assert result.reason == "TDX REPORT_DATA does not bind HCL runtime JSON"


def test_azure_tdx_vtpm_rejects_nonzero_report_data_tail():
    """The 32 REPORT_DATA bytes past the runtime-data hash must be zero.

    Azure fills only the first half. The tail here is set in the TDREPORT
    itself, so the HCL copy and the signed DCAP body agree and the quote
    signature verifies: only the tail rule can refuse it.
    """
    bundle, trust = _bundle(report_data_tail=b"\x01" * 32)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "unexpected nonzero TDX REPORT_DATA tail"


def test_azure_tdx_vtpm_rejects_unlinked_ak():
    bundle, trust = _bundle(wrong_ak=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM quote key does not match HCL-authenticated AK"


def test_azure_tdx_vtpm_rejects_runtime_json_without_hclakpub():
    bundle, trust = _bundle(
        runtime_json_override=json.dumps({"keys": [], "user-data": ""}).encode()
    )
    result = _verify(bundle, trust)
    assert not result.verified
    assert (result.reason or "").startswith("invalid HCLAkPub:")


# -- #173 HCLAkPub rules ---------------------------------------------------------


@pytest.mark.parametrize("member", ["d", "p", "q", "dp", "dq", "qi", "oth", "k"])
@pytest.mark.parametrize("value", ["AQAB", None], ids=["value", "null"])
def test_azure_tdx_vtpm_rejects_private_jwk_member(member, value):
    ak_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    runtime = _runtime_json(ak_key.public_key().public_numbers().n)
    bundle, trust = _bundle(ak_key=ak_key, runtime_json_override=runtime)
    assert _verify(bundle, trust).verified
    doc = json.loads(runtime)
    doc["keys"][0][member] = [] if member == "oth" and value is not None else value
    # Preserve present-null members and sign the altered runtime data.
    bundle, trust = _bundle(ak_key=ak_key, runtime_json_override=json.dumps(doc).encode())
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == (
        f"invalid HCLAkPub: HCLAkPub must not contain private key material ({member})"
    )


def test_azure_tdx_vtpm_rejects_hcl_ak_with_non_default_exponent():
    bundle, trust = _bundle(jwk={"e": "Aw"})
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "HCLAkPub exponent is not 65537"


def test_azure_tdx_vtpm_rejects_short_hcl_ak_modulus():
    short = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    bundle, trust = _bundle(jwk={"n": _b64url(short.public_key().public_numbers().n, 128)})
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "HCLAkPub modulus is shorter than 2048 bits"


def test_azure_tdx_vtpm_rejects_malformed_hcl_ak_jwk():
    cases = {
        "missing e": {"e": None},
        "missing kty": {"kty": None},
        "not RSA": {"kty": "EC"},
        "padded n": {"n": "AQAB=="},
        "padded e": {"e": "AQAB="},
        "standard alphabet": {"e": "AQ+B"},
        "e not a string": {"e": 65537},
    }
    for label, change in cases.items():
        bundle, trust = _bundle(jwk=change)
        result = _verify(bundle, trust)
        assert not result.verified, label
        assert (result.reason or "").startswith("invalid HCLAkPub:"), (label, result.reason)


# The AK is the HCL-authenticated key, modulus AND exponent. Matching only the
# modulus let an attacker-supplied ak_pem choose the exponent (#173).


def _supplied_ak_n(bundle: str) -> int:
    doc = json.loads(base64.b64decode(bundle))
    return serialization.load_pem_public_key(doc["ak_pem"].encode()).public_numbers().n


def _forged_e1_bundle(bundle: str, nonce: str, binding: bytes, measurement: str) -> str:
    """Under e=1 a PKCS#1 v1.5 signature is just the padded digest."""
    quote = (
        b"\xffTCG\x80\x18" + _tpm2b(b"signer")
        + _tpm2b(hashlib.sha256(bytes.fromhex(nonce) + binding).digest())
        + bytes(25) + (1).to_bytes(4, "big") + b"\x00\x0b\x03\x00\x00\x80"
        + _tpm2b(expected_pcr23_digest(measurement))
    )
    t = _SHA256_DIGEST_INFO + hashlib.sha256(quote).digest()
    forged = b"\x00\x01" + b"\xff" * (256 - len(t) - 3) + b"\x00" + t
    return _redo(
        bundle,
        ak_pem=_rsa_spki_pem(_supplied_ak_n(bundle), 1),
        tpm_quote_b64=base64.b64encode(quote).decode(),
        tpm_signature_b64=base64.b64encode(b"\x00\x14\x00\x0b" + _tpm2b(forged)).decode(),
    )


def test_azure_tdx_vtpm_rejects_exponent_one_forgery():
    bundle, trust = _bundle()
    nonce, binding, measurement = "11" * 32, b"\x22" * 32, "sha256:" + "ee" * 32
    evil = _forged_e1_bundle(bundle, nonce, binding, measurement)
    result = _verifier(trust).verify(
        evil, expected_nonce=nonce, channel_binding=binding,
        expected_workload_measurement=measurement, now=TDX_NOW,
    )
    assert not result.verified
    # cryptography 50 refuses to load e=1 at all; older releases load it and
    # the HCL key comparison is what denies. Either way it must not verify.
    reason = result.reason or ""
    assert "does not match" in reason or "e must be" in reason, reason


class _LoadedAk:
    """An RSA public key as an older cryptography would load it from ak_pem.

    ``verify`` accepts anything, so a pass could only come from verifying under
    this sender-supplied object instead of the key built from HCLAkPub.
    """

    def __init__(self, n: int, e: int) -> None:
        self._numbers = rsa.RSAPublicNumbers(e, n)

    def public_numbers(self) -> rsa.RSAPublicNumbers:
        return self._numbers

    def verify(self, *args, **kwargs) -> None:
        return None


rsa.RSAPublicKey.register(_LoadedAk)


def _load_as(n: int, e: int):
    return lambda data, *args, **kwargs: _LoadedAk(n, e)


def test_azure_tdx_vtpm_verifies_under_the_hcl_key_not_ak_pem(monkeypatch):
    # ak_pem names the right (n, e) but is an object that accepts any
    # signature. The forged quote must still fail: the signature is checked
    # under the key built from HCLAkPub.
    bundle, trust = _bundle()
    nonce, binding, measurement = "11" * 32, b"\x22" * 32, "sha256:" + "ee" * 32
    evil = _forged_e1_bundle(bundle, nonce, binding, measurement)
    monkeypatch.setattr(
        azure_vtpm.serialization, "load_pem_public_key",
        _load_as(_supplied_ak_n(bundle), 65537),
    )
    result = _verifier(trust).verify(
        evil, expected_nonce=nonce, channel_binding=binding,
        expected_workload_measurement=measurement, now=TDX_NOW,
    )
    assert not result.verified
    assert (result.reason or "").startswith("TPM quote verification failed:")


def test_azure_tdx_vtpm_rejects_ak_pem_with_matching_modulus_other_exponent():
    bundle, trust = _bundle()
    evil = _redo(bundle, ak_pem=_rsa_spki_pem(_supplied_ak_n(bundle), 3))
    result = _verify(evil, trust)
    assert not result.verified
    assert result.reason == "TPM quote key does not match HCL-authenticated AK"


# -- the vTPM PCR 23 quote ----------------------------------------------------


@pytest.mark.parametrize(
    "nonce",
    ["AB" * 32, "ab " * 32, NONCE + "\n", "ab" * 31, "ab" * 33, "", 123, None, b"ab" * 32],
    ids=["uppercase", "whitespace", "newline", "short", "long", "empty", "int", "null", "bytes"],
)
def test_azure_tdx_vtpm_rejects_malformed_expected_nonce(nonce):
    bundle, trust = _bundle()
    assert _verify(bundle, trust, expected_nonce=NONCE).verified
    wrong = _verify(bundle, trust, expected_nonce="ef" * 32)
    assert not wrong.verified
    assert wrong.reason == "TPM qualifying data does not bind nonce and transport key"
    result = _verify(bundle, trust, expected_nonce=nonce)
    assert not result.verified
    assert result.reason == "expected_nonce must be 64 lowercase hexadecimal characters"


@pytest.mark.parametrize(
    "bundle_kwargs, quote_override, reason",
    [
        ({}, b"invalid", "TPM quote verification failed: not a TPM generated quote"),
        ({"no_pcr23": True}, None, "TPM quote does not select exactly SHA-256 PCR 23"),
        (
            {"pcr_digest": bytes(32)},
            None,
            "TPM PCR digest does not match the approved workload measurement",
        ),
    ],
    ids=["parsing", "selection", "pcr-digest"],
)
def test_azure_tdx_vtpm_checks_nonce_before_tpm_evidence(bundle_kwargs, quote_override, reason):
    bundle, trust = _bundle(**bundle_kwargs)
    if quote_override is not None:
        bundle = _redo(bundle, tpm_quote_b64=base64.b64encode(quote_override).decode())
    evidence_result = _verify(bundle, trust, expected_nonce=NONCE)
    assert not evidence_result.verified
    assert evidence_result.reason == reason
    result = _verify(bundle, trust, expected_nonce="AB" * 32)
    assert not result.verified
    assert result.reason == "expected_nonce must be 64 lowercase hexadecimal characters"


def test_azure_tdx_vtpm_rejects_wrong_nonce_binding():
    bundle, trust = _bundle(wrong_binding=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM qualifying data does not bind nonce and transport key"


def test_azure_tdx_vtpm_requires_pcr23():
    bundle, trust = _bundle(no_pcr23=True)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM quote does not select exactly SHA-256 PCR 23"


def test_azure_tdx_vtpm_rejects_changed_pcr_state():
    bundle, trust = _bundle(pcr_digest=b"\xff" * 32)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM PCR digest does not match the approved workload measurement"


def test_azure_tdx_vtpm_rejects_malformed_pcr_digest():
    bundle, trust = _bundle(pcr_digest=b"short")
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM quote verification failed: TPM PCR digest is not 32-byte SHA-256"


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


# -- public unwrap helper ------------------------------------------------------


def test_unwrap_azure_tdx_vtpm_bundle_round_trips_every_member():
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
        (
            base64.b64encode(b'{"kind": "wcm-azure-snp-vtpm/v1"}').decode(),
            "unexpected Azure TDX vTPM bundle kind",
        ),
        (
            base64.b64encode(b'{"kind": "wcm-azure-tdx-vtpm/v1", "hcl_b64": ""}').decode(),
            "invalid Azure vTPM evidence",
        ),
    ],
)
def test_unwrap_azure_tdx_vtpm_bundle_refuses_malformed_input(quote_b64, expected):
    with pytest.raises(ValueError, match=expected):
        unwrap_azure_tdx_vtpm_bundle(quote_b64)


# -- the MRTD allowlist and the HCL TDREPORT --------------------------------------


def test_td_whose_mrtd_is_not_on_the_allowlist_is_denied_when_all_else_passes():
    other_mrtd = b"\x44" * 48
    bundle, trust = _bundle(mrtd=other_mrtd)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TD MRTD is not an allowed Azure paravisor measurement"
    # The same bundle passes once its MRTD is on the allowlist, so the
    # allowlist is the check that decided.
    assert _verify(bundle, trust, measurements=(other_mrtd.hex(),)).verified


def test_accepts_any_listed_paravisor_measurement_as_hex_or_bytes():
    bundle, trust = _bundle()
    assert _verify(bundle, trust, measurements=("44" * 48, PARAVISOR)).verified
    assert _verify(bundle, trust, measurements=(MRTD,)).verified


@pytest.mark.parametrize(
    "measurements, error",
    [
        ((), ValueError),
        (set(), ValueError),
        (PARAVISOR, TypeError),
        (MRTD, TypeError),
        (None, TypeError),
        ("ab" * 48, TypeError),
        (("AB" * 48,), ValueError),
        (("33" * 32,), ValueError),
        (("zz" * 48,), ValueError),
        (("sha384:" + "33" * 48,), ValueError),
        ((48,), TypeError),
        ((bytearray(48),), TypeError),
        ((__import__("collections").UserString(PARAVISOR),), TypeError),
    ],
)
def test_requires_a_valid_paravisor_allowlist(measurements, error):
    with pytest.raises(error):
        AzureTdxVtpmVerifier(TrustStore(), paravisor_measurements=measurements)


def test_has_no_default_allowlist():
    with pytest.raises(TypeError):
        AzureTdxVtpmVerifier(TrustStore())  # type: ignore[call-arg]


def test_success_carries_no_snp_report():
    bundle, trust = _bundle()
    result = _verify(bundle, trust)
    assert result.verified and result.snp_report is None


@pytest.mark.parametrize("field", [f[0] for f in TDREPORT_FIELDS])
def test_hcl_tdreport_that_differs_from_the_signed_body_is_denied(field):
    _, _, length, tdreport_offset = next(f for f in TDREPORT_FIELDS if f[0] == field)
    bundle, trust = _bundle(hcl_tdreport_change=(tdreport_offset + length - 1, 0x01))
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == f"HCL TD report does not match the Intel-signed TD report ({field})"


def test_hcl_tdreport_mac_is_not_compared():
    # The REPORTMACSTRUCT MAC has no counterpart in the DCAP body and can only
    # be checked on the platform; altering it is not a mismatch.
    bundle, trust = _bundle(hcl_tdreport_change=(224, 0x01))
    assert _verify(bundle, trust).verified


def test_debug_td_is_denied():
    bundle, trust = _bundle(td_attributes=0x10000001)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == "TDX TD attributes allow debugging"


# -- HCL framing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "hcl_kwargs, reason",
    [
        ({"report_size_delta": 1}, "HCL report size does not match its runtime data"),
        ({"report_size_delta": -1}, "HCL report size does not match its runtime data"),
        ({"data_size_delta": 1}, "HCL runtime data size does not match its JSON size"),
        (
            {"report_type": 2},
            "HCL runtime data report type 2 differs from the value observed on Azure (4)",
        ),
        (
            {"hash_type": 2},
            "HCL runtime data hash type 2 differs from the value observed on Azure (1)",
        ),
        (
            {"runtime_version": 2},
            "HCL runtime data version 2 differs from the value observed on Azure (1)",
        ),
        (
            {"header_version": 1},
            "HCL header version 1 differs from the value observed on Azure (2)",
        ),
        (
            {"request_type": 1},
            "HCL header request type 1 differs from the value observed on Azure (2)",
        ),
    ],
)
def test_inconsistent_hcl_framing_is_denied(hcl_kwargs, reason):
    bundle, trust = _bundle(hcl_kwargs=hcl_kwargs)
    result = _verify(bundle, trust)
    assert not result.verified and result.reason == reason


def test_hcl_without_padding_is_accepted():
    bundle, trust = _bundle(hcl_kwargs={"padding_len": 0})
    assert _verify(bundle, trust).verified


def test_nonzero_hcl_padding_is_denied():
    bundle, trust = _bundle()
    doc = json.loads(base64.b64decode(bundle))
    hcl = bytearray(base64.b64decode(doc["hcl_b64"]))
    hcl[-1] = 0x01
    result = _verify(_redo(bundle, hcl_b64=base64.b64encode(bytes(hcl)).decode()), trust)
    assert not result.verified and result.reason == "HCL has nonzero bytes after the report"


def test_nonzero_hcl_report_slot_tail_is_denied():
    bundle, trust = _bundle()
    doc = json.loads(base64.b64decode(bundle))
    hcl = bytearray(base64.b64decode(doc["hcl_b64"]))
    hcl[_REPORT_OFFSET + 1024] = 0x01
    result = _verify(_redo(bundle, hcl_b64=base64.b64encode(bytes(hcl)).decode()), trust)
    assert not result.verified
    assert result.reason == "HCL report slot has nonzero bytes after the TD report"


# -- PCR selection: exactly SHA-256 PCR 23 (TDX path) -----------------------------


_DIGEST = expected_pcr23_digest(MEASUREMENT)


@pytest.mark.parametrize(
    "selections, count",
    [
        # SHA-256 PCR 23 plus a SHA-1 bank selection.
        (b"\x00\x0b\x03\x00\x00\x80" + b"\x00\x04\x03\x01\x00\x00", 2),
        # SHA-256 with a 4-byte bitmap selecting PCR 23.
        (b"\x00\x0b\x04\x00\x00\x80\x00", 1),
        # SHA-256 PCR 23 and PCR 16.
        (b"\x00\x0b\x03\x00\x00\x81", 1),
    ],
)
def test_tdx_path_requires_exactly_sha256_pcr23(selections, count):
    # Signed by the real AK over the approved digest, so signature, digest and
    # nonce all hold; only the TDX path's exact-selection rule refuses them.
    bundle, trust = _bundle(selections=selections, selection_count=count, pcr_digest=_DIGEST)
    result = _verify(bundle, trust)
    assert not result.verified
    assert result.reason == "TPM quote does not select exactly SHA-256 PCR 23"
