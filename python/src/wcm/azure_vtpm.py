"""Fail-closed verification of Azure SEV-SNP and Intel TDX plus fresh vTPM evidence."""
from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Union

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from ._quote_verify import (
    QuoteFormatError,
    QuoteVerification,
    TrustStore,
    _nonce_bytes,
    _unique_object,
    verify_cert_chain,
)
from ._certificates import load_pem_certificate
from .snp import extract_snp_report_from_hcl, parse_snp_report, verify_snp_report_signature
from .tdx import TD_ATTR_DEBUG, parse_tdx_quote, verify_tdx_quote

_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+\Z")
_JWK_PRIVATE_PARAMS = frozenset({"d", "p", "q", "dp", "dq", "qi", "oth", "k"})


def expected_pcr23_digest(measurement: str) -> bytes:
    """Return a quote's pcrDigest for measured-launch SHA-256 PCR 23.

    The launch agent resets the application-owned PCR 23 to zero and extends
    the manifest's 32-byte event digest once. TPM extend semantics produce the
    PCR value as ``SHA256(old_pcr || event_digest)``. ``TPMS_QUOTE_INFO`` then
    carries the SHA-256 digest of the concatenated selected PCR values. Because
    WCM selects only PCR 23, the signed value is ``SHA256(pcr23_value)``.
    """
    algorithm, separator, digest = measurement.partition(":")
    if separator != ":" or algorithm != "sha256" or len(digest) != 64:
        raise ValueError("workload measurement must be sha256:<64 lowercase hex digits>")
    if digest.lower() != digest:
        raise ValueError("workload measurement hex must be lowercase")
    try:
        event_digest = bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("workload measurement contains non-hex characters") from exc
    pcr23_value = hashlib.sha256(bytes(32) + event_digest).digest()
    return hashlib.sha256(pcr23_value).digest()


def _b64(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


# Azure vTPM AKs are RSA-2048 with the TPM default exponent; HCLAkPub carries
# e = "AQAB" on every capture WCM has seen.
_AK_EXPONENT = 65537
_AK_MIN_BITS = 2048


def _b64url_uint(value: str) -> int:
    """Decode a JWK Base64urlUInt (RFC 7518 section 2), rejecting non-minimal encodings."""
    if not isinstance(value, str) or not _B64URL_RE.match(value) or len(value) % 4 == 1:
        raise ValueError("JWK integer must be an unpadded base64url string")
    std = value.replace("-", "+").replace("_", "/")
    raw = base64.b64decode(std + "=" * (-len(std) % 4), validate=True)
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value:
        raise ValueError("JWK integer is not canonical base64url")
    if not raw or raw[0] == 0:
        raise ValueError("JWK Base64urlUInt must not be empty or carry leading zero bytes")
    return int.from_bytes(raw, "big")


_MEASUREMENT_LEN = 48
_POLICY_DEBUG = 1 << 19


def _paravisor_measurement(value: Union[str, bytes]) -> bytes:
    """Normalise one allowed launch measurement (SNP MEASUREMENT or TDX MRTD) to 48 raw bytes."""
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        if value != value.lower():
            raise ValueError("paravisor measurement hex must be lowercase")
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("paravisor measurement contains non-hex characters") from exc
    else:
        raise TypeError("paravisor measurement must be hex str or bytes")
    if len(raw) != _MEASUREMENT_LEN:
        raise ValueError(f"paravisor measurement must be {_MEASUREMENT_LEN} bytes (SHA-384)")
    return raw


def _take_u16(blob: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 2 > len(blob):
        raise ValueError("truncated TPM2B")
    size = int.from_bytes(blob[offset : offset + 2], "big")
    start = offset + 2
    end = start + size
    if end > len(blob):
        raise ValueError("truncated TPM2B payload")
    return blob[start:end], end


class AzureSnpVtpmVerifier:
    """Verify SNP authenticity, the HCL AK link, and a fresh AK-signed TPM quote.

    ``quote_b64`` is base64 JSON containing ``hcl_b64``, ``ak_pem``,
    ``tpm_quote_b64``, ``tpm_signature_b64``, ``vcek_pem``,
    ``intermediates_pem`` and ``root_pem``. The TPM quote must use
    SHA-256(nonce || channel_binding) as qualifying data and select PCR 23.

    ``paravisor_measurements`` is the operator's allowlist of SNP launch
    measurements for the Azure paravisor builds it trusts. The REPORT_DATA
    binding below proves only that whoever requested the SNP report chose the
    HCL runtime data; any SNP guest that controls REPORT_DATA can do that. The
    launch measurement is what ties the report to Azure's paravisor rather than
    to an arbitrary guest, so it is required and there is no default. Azure
    updates the paravisor, so this is an allowlist the operator maintains, not
    a constant.
    """

    def __init__(
        self,
        trust_store: TrustStore,
        *,
        paravisor_measurements: Iterable[Union[str, bytes]],
    ) -> None:
        if isinstance(paravisor_measurements, (str, bytes)):
            raise TypeError("paravisor_measurements must be a collection, not one value")
        allowed = frozenset(_paravisor_measurement(m) for m in paravisor_measurements)
        if not allowed:
            raise ValueError("at least one Azure paravisor launch measurement is required")
        self._trust = trust_store
        self._paravisor_measurements = allowed

    def verify(
        self,
        quote_b64: str,
        *,
        expected_nonce: str,
        channel_binding: bytes = b"",
        expected_workload_measurement: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> QuoteVerification:
        try:
            doc: dict[str, Any] = json.loads(_b64(quote_b64), object_pairs_hook=_unique_object)
            if not isinstance(doc, dict):
                raise ValueError("quote bundle must be a JSON object")
            hcl = _b64(doc["hcl_b64"])
            quote = _b64(doc["tpm_quote_b64"])
            signature_blob = _b64(doc["tpm_signature_b64"])
            ak = serialization.load_pem_public_key(doc["ak_pem"].encode())
            # Azure's THIM currently serves an otherwise verifiable AMD VCEK with
            # a non-positive serial. Keep the compatibility exception local to
            # this authenticated provider leaf; every other provider certificate
            # continues to use WCM's positive-serial policy.
            vcek = load_pem_certificate(
                doc["vcek_pem"].encode(), allow_non_positive_serial=True
            )
            intermediates = [
                load_pem_certificate(p.encode())
                for p in doc["intermediates_pem"]
            ]
            report = extract_snp_report_from_hcl(hcl)
            parsed = parse_snp_report(report)
        except (
            KeyError, ValueError, TypeError, AttributeError,
            UnsupportedAlgorithm, QuoteFormatError,
        ) as exc:
            # The bundle is attacker-supplied: every malformed shape is a
            # denial with a reason, never an exception out of the release path.
            return QuoteVerification(False, f"invalid Azure vTPM evidence: {exc}")
        if not isinstance(ak, rsa.RSAPublicKey):
            return QuoteVerification(False, "HCL AK is not RSA")
        current = now or datetime.now(timezone.utc)
        chain_error = verify_cert_chain(vcek, intermediates, self._trust, current)
        if chain_error:
            return QuoteVerification(False, chain_error)
        if not verify_snp_report_signature(report, vcek):
            return QuoteVerification(False, "SNP report signature does not verify")
        if parsed.measurement not in self._paravisor_measurements:
            return QuoteVerification(
                False, "SNP launch measurement is not an allowed Azure paravisor measurement"
            )
        if parsed.policy & _POLICY_DEBUG:
            return QuoteVerification(False, "SNP guest policy allows debugging")
        # The paravisor runs at VMPL 0 and requests this report itself; a report
        # requested at VMPL 1 to 3 came from less privileged guest software.
        if parsed.vmpl != 0:
            return QuoteVerification(False, f"SNP report is from VMPL {parsed.vmpl}, not VMPL 0")

        runtime = hcl[32 + 1184 :]
        if len(runtime) < 20:
            return QuoteVerification(False, "HCL runtime data is truncated")
        json_len = int.from_bytes(runtime[16:20], "little")
        runtime_json = runtime[20 : 20 + json_len]
        if len(runtime_json) != json_len:
            return QuoteVerification(False, "HCL runtime JSON is truncated")
        if parsed.report_data[:32] != hashlib.sha256(runtime_json).digest():
            return QuoteVerification(False, "SNP REPORT_DATA does not bind HCL runtime JSON")
        if parsed.report_data[32:] != bytes(32):
            return QuoteVerification(False, "unexpected nonzero SNP REPORT_DATA tail")
        try:
            runtime_doc = json.loads(runtime_json, object_pairs_hook=_unique_object)
            if not isinstance(runtime_doc, dict):
                raise ValueError("HCL runtime JSON must be a JSON object")
            jwk = next(
                k for k in runtime_doc["keys"]
                if isinstance(k, dict) and k.get("kid") == "HCLAkPub"
            )
            private_members = sorted(_JWK_PRIVATE_PARAMS.intersection(jwk))
            if private_members:
                raise ValueError(
                    f"HCLAkPub must not contain private key material ({', '.join(private_members)})"
                )
            if jwk["kty"] != "RSA":
                raise ValueError(f"kty {jwk['kty']!r} is not RSA")
            modulus = _b64url_uint(jwk["n"])
            exponent = _b64url_uint(jwk["e"])
        except (KeyError, StopIteration, ValueError, TypeError, AttributeError) as exc:
            return QuoteVerification(False, f"invalid HCLAkPub: {exc}")
        if exponent != _AK_EXPONENT:
            return QuoteVerification(False, f"HCLAkPub exponent {exponent} is not {_AK_EXPONENT}")
        if modulus.bit_length() < _AK_MIN_BITS:
            return QuoteVerification(False, f"HCLAkPub modulus is shorter than {_AK_MIN_BITS} bits")
        # The quote is verified under the key the SNP report authenticates,
        # never under ``ak_pem``: that field is attacker-supplied, and matching
        # only its modulus let an (n, e=1) key verify a forged signature.
        # ``ak_pem`` must still name exactly the same key.
        supplied = ak.public_numbers()
        if (supplied.n, supplied.e) != (modulus, exponent):
            return QuoteVerification(False, "TPM quote key does not match HCL-authenticated AK")
        ak = rsa.RSAPublicNumbers(exponent, modulus).public_key()

        try:
            if quote[:4] != b"\xffTCG" or quote[4:6] != b"\x80\x18":
                raise ValueError("not a TPM generated quote")
            _, offset = _take_u16(quote, 6)  # qualified signer
            extra_data, offset = _take_u16(quote, offset)
            offset += 25  # TPMS_CLOCK_INFO (17) + firmwareVersion (8)
            if offset + 4 > len(quote):
                raise ValueError("truncated TPM PCR selection")
            count = int.from_bytes(quote[offset : offset + 4], "big")
            offset += 4
            selected_pcr23 = False
            for _ in range(count):
                if offset + 3 > len(quote):
                    raise ValueError("truncated TPM PCR selection")
                algorithm = int.from_bytes(quote[offset : offset + 2], "big")
                size = quote[offset + 2]
                selection = quote[offset + 3 : offset + 3 + size]
                if len(selection) != size:
                    raise ValueError("truncated TPM PCR selection")
                offset += 3 + size
                selected_pcr23 |= algorithm == 0x000B and size >= 3 and bool(selection[2] & 0x80)
            if not selected_pcr23:
                return QuoteVerification(False, "TPM quote does not select SHA-256 PCR 23")
            pcr_digest, offset = _take_u16(quote, offset)
            if offset != len(quote):
                raise ValueError("trailing TPM quote bytes")
            if len(pcr_digest) != hashlib.sha256().digest_size:
                raise ValueError("TPM PCR digest is not 32-byte SHA-256")
            if expected_workload_measurement is None:
                return QuoteVerification(
                    False, "expected workload measurement is required by release policy"
                )
            if pcr_digest != expected_pcr23_digest(expected_workload_measurement):
                return QuoteVerification(
                    False, "TPM PCR digest does not match the approved workload measurement"
                )
            if extra_data != hashlib.sha256(_nonce_bytes(expected_nonce) + channel_binding).digest():
                return QuoteVerification(False, "TPM qualifying data does not bind nonce and transport key")
            if signature_blob[:4] != b"\x00\x14\x00\x0b":
                raise ValueError("TPM signature is not RSASSA/SHA-256")
            signature, end = _take_u16(signature_blob, 4)
            if end != len(signature_blob):
                raise ValueError("trailing TPM signature bytes")
            ak.verify(signature, quote, padding.PKCS1v15(), hashes.SHA256())
        except (ValueError, InvalidSignature) as exc:
            return QuoteVerification(False, f"TPM quote verification failed: {exc}")
        return QuoteVerification(
            True, leaf_subject=vcek.subject.rfc4514_string(), snp_report=report
        )


# -- Azure Intel TDX ----------------------------------------------------------
#
# Everything below is used by AzureTdxVtpmVerifier only; AzureSnpVtpmVerifier
# above keeps its own inline checks.

TDX_VTPM_BUNDLE_KIND = "wcm-azure-tdx-vtpm/v1"

_HCL_MAGIC = b"HCLA"
_HCL_REPORT_OFFSET = 32
# REPORTMACSTRUCT TYPE byte of a TDX TD report, at the HCL report offset.
_HCL_TD_REPORT_TYPE = 0x81


@dataclass(frozen=True)
class AzureTdxVtpmBundle:
    """The decoded members of a ``wcm-azure-tdx-vtpm/v1`` bundle.

    Decoding authenticates nothing: every member is as the sender wrote it
    until ``AzureTdxVtpmVerifier`` has checked it.
    """

    dcap_quote: bytes
    hcl: bytes
    ak_pem: str
    tpm_quote: bytes
    tpm_signature: bytes


def unwrap_azure_tdx_vtpm_bundle(quote_b64: str) -> AzureTdxVtpmBundle:
    """Decode the ``wcm-azure-tdx-vtpm/v1`` bundle carried in ``CpuQuote.quote_b64``.

    The one place that knows the bundle's wire shape: base64 of a JSON object
    with ``kind``, ``tdx_quote_b64``, ``hcl_b64``, ``ak_pem``, ``tpm_quote_b64``
    and ``tpm_signature_b64``. Tools that need only the DCAP quote (MRTD
    extraction, ``wcm verify-quote --kind tdx``) call this instead of parsing
    the JSON themselves. It verifies nothing.

    Raises ``ValueError`` on any shape problem: not strict base64, not JSON,
    not an object, the wrong ``kind``, or a missing or undecodable member.
    """
    try:
        doc = json.loads(_b64(quote_b64))
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"invalid Azure vTPM evidence: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("invalid Azure vTPM evidence: bundle is not a JSON object")
    kind = doc.get("kind")
    if kind != TDX_VTPM_BUNDLE_KIND:
        raise ValueError(f"unexpected Azure TDX vTPM bundle kind: {kind!r}")
    try:
        ak_pem = doc["ak_pem"]
        if not isinstance(ak_pem, str):
            raise TypeError("ak_pem is not a string")
        return AzureTdxVtpmBundle(
            dcap_quote=_b64(doc["tdx_quote_b64"]),
            hcl=_b64(doc["hcl_b64"]),
            ak_pem=ak_pem,
            tpm_quote=_b64(doc["tpm_quote_b64"]),
            tpm_signature=_b64(doc["tpm_signature_b64"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid Azure vTPM evidence: {exc}") from exc


# HCL framing, from the bytes of the 2026-09-16 Standard_DC4es_v6 capture
# (2600-byte NV blob): a 32-byte header of little-endian u32 (magic, version 2,
# report size 2466 = 32 + 1184 + 1250, request type 2, then zeros); a
# 1184-byte hardware-report slot holding the 1024-byte TDREPORT and 160 zero
# bytes; a 20-byte runtime-data header of five u32 (data size 1250 = 20 +
# JSON size, version 1, report type 4, report-data hash type 1, JSON size
# 1230); the JSON; then zero padding to the end of the NV blob (134 bytes).
_HCL_VERSION_OFFSET = 4
_HCL_REPORT_SIZE_OFFSET = 8
_HCL_REQUEST_TYPE_OFFSET = 12
_HCL_SLOT_LEN = 1184
_TDREPORT_LEN = 1024
_HCL_RUNTIME_HEADER_LEN = 20
# Header values observed on Azure TDX captures, the only evidence for them,
# so anything else is refused rather than interpreted. The hash type (1) is
# read as the SHA-256 the verifier recomputes over the JSON. Within one
# allowlisted MRTD the paravisor code that writes these is fixed.
_HCL_OBSERVED = (
    # (name, offset, value)
    ("HCL header version", _HCL_VERSION_OFFSET, 2),
    ("HCL header request type", _HCL_REQUEST_TYPE_OFFSET, 2),
    ("HCL runtime data version", _HCL_REPORT_OFFSET + _HCL_SLOT_LEN + 4, 1),
    ("HCL runtime data report type", _HCL_REPORT_OFFSET + _HCL_SLOT_LEN + 8, 4),
    ("HCL runtime data hash type", _HCL_REPORT_OFFSET + _HCL_SLOT_LEN + 12, 1),
)

# Where each field of the Intel-signed DCAP v4 TD report body (584 bytes)
# sits in the 1024-byte TDREPORT_STRUCT the HCL carries: REPORTMACSTRUCT
# (256 bytes, REPORTDATA at 128), TEE_TCB_INFO (239 bytes at 256: VALID 8,
# TEE_TCB_SVN 16, MRSEAM 48, MRSIGNERSEAM 48, ATTRIBUTES 8, ...), 17 reserved
# bytes, then TDINFO (512 bytes at 512: ATTRIBUTES 8, XFAM 8, MRTD 48,
# MRCONFIGID 48, MROWNER 48, MROWNERCONFIG 48, RTMR0-3 192, ...). The twelve
# ranges cover all 584 body bytes, so the body is a pure function of the
# TDREPORT; every range was checked equal on the capture's real bytes.
# (name, offset in DCAP body, length, offset in TDREPORT)
_TDREPORT_TO_DCAP_BODY = (
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


def _u32le(blob: bytes, offset: int) -> int:
    return int.from_bytes(blob[offset : offset + 4], "little")


def _tdx_hcl_framing_error(hcl: bytes) -> Optional[str]:
    """Return why the HCL's sizes and padding are inconsistent, or None.

    Checked before anything reads the HCL at fixed offsets.
    """
    runtime_header = _HCL_REPORT_OFFSET + _HCL_SLOT_LEN
    if len(hcl) < runtime_header + _HCL_RUNTIME_HEADER_LEN:
        return "HCL runtime data is truncated"
    report_size = _u32le(hcl, _HCL_REPORT_SIZE_OFFSET)
    data_size = _u32le(hcl, runtime_header)
    json_size = _u32le(hcl, runtime_header + 16)
    if data_size != _HCL_RUNTIME_HEADER_LEN + json_size:
        return "HCL runtime data size does not match its JSON size"
    if report_size != runtime_header + data_size:
        return "HCL report size does not match its runtime data"
    if report_size > len(hcl):
        return "HCL report is truncated"
    if any(hcl[report_size:]):
        return "HCL has nonzero bytes after the report"
    if any(hcl[_HCL_REPORT_OFFSET + _TDREPORT_LEN : runtime_header]):
        return "HCL report slot has nonzero bytes after the TD report"
    for name, offset, observed in _HCL_OBSERVED:
        value = _u32le(hcl, offset)
        if value != observed:
            return f"{name} {value} differs from the value observed on Azure ({observed})"
    return None


def _tdreport_mismatch(tdreport: bytes, signed_body: bytes) -> Optional[str]:
    """Name the first DCAP body field the HCL's TDREPORT disagrees on, or None."""
    for name, body_offset, length, tdreport_offset in _TDREPORT_TO_DCAP_BODY:
        if (
            tdreport[tdreport_offset : tdreport_offset + length]
            != signed_body[body_offset : body_offset + length]
        ):
            return name
    return None


# TPMS_ATTEST PCR selection the TDX path accepts: count 1, SHA-256, a 3-byte
# bitmap with only PCR 23 set.
_PCR23_ONLY_SELECTION = (1).to_bytes(4, "big") + b"\x00\x0b\x03\x00\x00\x80"


def _tdx_hcl_authenticated_ak(
    runtime_json: bytes, supplied_ak: rsa.RSAPublicKey
) -> tuple[Optional[rsa.RSAPublicKey], Optional[str]]:
    """Build the TPM quote key from ``HCLAkPub``, or return ``(None, reason)``.

    Call only after the signed REPORT_DATA has been checked against
    ``sha256(runtime_json)``. The rules are #173's: ``kty`` RSA, unpadded
    base64url ``n`` and ``e``, ``e`` = 65537, a modulus of at least 2048
    bits, and ``ak_pem`` naming exactly the same key; the key returned is
    built from ``HCLAkPub``, never taken from ``ak_pem``.
    """
    try:
        runtime_doc = json.loads(runtime_json)
        jwk = next(k for k in runtime_doc["keys"] if k["kid"] == "HCLAkPub")
        if jwk["kty"] != "RSA":
            raise ValueError(f"kty {jwk['kty']!r} is not RSA")
        modulus = _b64url_uint(jwk["n"])
        exponent = _b64url_uint(jwk["e"])
    except (KeyError, StopIteration, ValueError, TypeError, AttributeError, RecursionError) as exc:
        return None, f"invalid HCLAkPub: {exc}"
    if exponent != _AK_EXPONENT:
        return None, f"HCLAkPub exponent is not {_AK_EXPONENT}"
    if modulus.bit_length() < _AK_MIN_BITS:
        return None, f"HCLAkPub modulus is shorter than {_AK_MIN_BITS} bits"
    supplied = supplied_ak.public_numbers()
    if (supplied.n, supplied.e) != (modulus, exponent):
        return None, "TPM quote key does not match HCL-authenticated AK"
    return rsa.RSAPublicNumbers(exponent, modulus).public_key(), None


def _tdx_verify_pcr23_quote(
    quote: bytes,
    signature_blob: bytes,
    ak: rsa.RSAPublicKey,
    *,
    expected_nonce: str,
    channel_binding: bytes,
    expected_workload_measurement: Optional[str],
) -> Optional[str]:
    """Return a refusal reason for the AK-signed PCR 23 quote, or None if it holds.

    ``quote`` is a ``TPMS_ATTEST`` and ``signature_blob`` a ``TPMT_SIGNATURE``.
    Checks, in order: the quote is TPM-generated; it selects exactly SHA-256
    PCR 23 and nothing else (a TPM's pcrDigest covers every selected PCR, so
    this narrows what is accepted rather than closing a forgery); its signed
    ``pcrDigest`` is the measured-launch value for the approved workload
    measurement; its qualifying data is ``sha256(nonce || channel_binding)``;
    and ``ak`` verifies the RSASSA-PKCS1-v1_5/SHA-256 signature over the quote
    bytes. Malformed evidence is a reason, never an exception.
    """
    try:
        if quote[:4] != b"\xffTCG" or quote[4:6] != b"\x80\x18":
            raise ValueError("not a TPM generated quote")
        _, offset = _take_u16(quote, 6)  # qualified signer
        extra_data, offset = _take_u16(quote, offset)
        offset += 25  # TPMS_CLOCK_INFO (17) + firmwareVersion (8)
        if quote[offset : offset + len(_PCR23_ONLY_SELECTION)] != _PCR23_ONLY_SELECTION:
            return "TPM quote does not select exactly SHA-256 PCR 23"
        pcr_digest, offset = _take_u16(quote, offset + len(_PCR23_ONLY_SELECTION))
        if offset != len(quote):
            raise ValueError("trailing TPM quote bytes")
        if len(pcr_digest) != hashlib.sha256().digest_size:
            raise ValueError("TPM PCR digest is not 32-byte SHA-256")
        if expected_workload_measurement is None:
            return "expected workload measurement is required by release policy"
        if pcr_digest != expected_pcr23_digest(expected_workload_measurement):
            return "TPM PCR digest does not match the approved workload measurement"
        if extra_data != hashlib.sha256(bytes.fromhex(expected_nonce) + channel_binding).digest():
            return "TPM qualifying data does not bind nonce and transport key"
        if signature_blob[:4] != b"\x00\x14\x00\x0b":
            raise ValueError("TPM signature is not RSASSA/SHA-256")
        signature, end = _take_u16(signature_blob, 4)
        if end != len(signature_blob):
            raise ValueError("trailing TPM signature bytes")
        ak.verify(signature, quote, padding.PKCS1v15(), hashes.SHA256())
    except (ValueError, InvalidSignature) as exc:
        return f"TPM quote verification failed: {exc}"
    return None


class AzureTdxVtpmVerifier:
    """Verify an Azure Intel TDX DCAP quote plus fresh AK-signed vTPM evidence.

    This verifier accepts only the bundle produced by
    ``AzureTdxVtpmProvider(vtpm_bundle=True)``. The provider's default raw DCAP
    output remains suitable for ``verify_tdx_quote`` with
    ``expected_nonce=None``, but has no caller-nonce freshness binding.

    PROVISIONAL. The hardware evidence for it is nine bundles from Azure
    DCesv6 TDs in westus3: one from a fork build of ``AzureTdxVtpmProvider``
    on 2026-09-16, and eight from earlier builds of that provider on
    2026-10-16 and 2026-10-21. This verifier accepts all nine offline. No
    hardware run used this exact revision, and one person made and checked
    the captures. They are not committed (their signed claims carry the VM's
    unique ID and vTPM endorsement key); the tests use synthetic bundles.

    ``quote_b64`` is a ``wcm-azure-tdx-vtpm/v1`` bundle (see
    ``unwrap_azure_tdx_vtpm_bundle``). The TPM quote must select exactly
    SHA-256 PCR 23 and carry ``sha256(nonce || channel_binding)`` as
    qualifying data.

    ``paravisor_measurements`` is the operator's allowlist of TD launch
    measurements (MRTD) for the Azure paravisor builds it trusts, with the
    same rules as ``AzureSnpVtpmVerifier``'s: 48 raw bytes or 96 lowercase hex
    characters each, required, no default, and an empty set is refused at
    construction. A pass means an Intel-attested TD whose launch measurement
    is on that allowlist; the HCL's claim about the attestation key is that
    image's. MRTD identifies the launched image, not the host or the
    operator: the same image launched on another TDX host has the same MRTD,
    and on every capture so far MRCONFIGID, MROWNER and MROWNERCONFIG are all
    zero. MRTD can differ between captures (Azure TDX
    captures have shown different MRTDs), so this is an allowlist the operator maintains, not
    a constant.

    Checks, in order: the bundle shape and an RSA
    ``ak_pem``; the HCL magic and TD report type byte; the DCAP quote, with
    every check ``verify_tdx_quote`` makes (Intel TD QE identity, PCK chain to
    a root in the trust store) but ``expected_nonce=None``, because
    REPORT_DATA carries the runtime-data hash, not a nonce; the HCL's sizes,
    padding and five of its header values (the header's last 16 bytes are
    not checked); that every field of the Intel-signed TD report
    body equals the corresponding field of the TDREPORT in the HCL; that MRTD
    is allowed; that the TD attributes do not allow debugging; that the signed
    REPORT_DATA is the SHA-256 of the HCL runtime JSON with a zero tail; that
    ``HCLAkPub`` in that JSON is an acceptable RSA key which ``ak_pem`` names
    exactly; and that a TPM quote verified under the key built from
    ``HCLAkPub`` (never under ``ak_pem``) selects exactly SHA-256 PCR 23,
    carries the approved measured-launch digest and binds the nonce and the
    transport key. A malformed bundle JSON, HCL, runtime JSON, ``ak_pem``,
    TPMS_ATTEST or TPMT_SIGNATURE is a denial, never an exception. The DCAP
    quote itself is checked by ``verify_tdx_quote``.

    What a pass does not say:

    - Anything about the host or operator beyond the allowed MRTD.
    - Freshness beyond whoever issued ``expected_nonce``: the quote binds the
      nonce the producer was handed, fresh to a relying party only if that
      party's own ``Challenge`` minted it.
    - That the DCAP quote or the HCL is fresh: only the AK-signed TPM quote
      binds the challenge; the DCAP quote and the HCL this provider produces
      do not. The DCAP quote and the HCL show which TD the AK
      belongs to, not when they, or the MRTD and TCB they report, were
      produced.
    - That the approved image is the code running: PCR 23 binds the digest
      the guest supplied. Root in the guest can reset PCR 23 and extend it
      with the approved digest.
    - The AK-to-TD link is the claim of the allowlisted image, signed by
      Intel; that image is a trusted component.
    - That the guest could not choose the attestation key. TDX has no VMPL,
      so nothing in the evidence shows which software in the TD requested
      the report. In the public OpenHCL and Intel TDX-module sources, guest
      software cannot request a TD report itself; it can make the paravisor
      issue one and supply its 64-byte user-data, but the attestation key
      in the runtime claims is the paravisor's. Those sources are not tied
      to the MRTDs observed, so the allowlist carries this assumption.
    - Expected values for TD report fields other than MRTD and the DEBUG
      attribute. MRSIGNERSEAM, SEAMATTRIBUTES, the other TD attributes, XFAM,
      MRCONFIGID, MROWNER, MROWNERCONFIG and the RTMRs are compared with the
      signed body but not pinned.
    - The platform the evidence claims. The result covers the bundle, not the
      evidence's ``CpuQuote.platform`` label, which a key broker reads for
      ``cpu_platform_allowed``; making sure the manifest's required platform
      matches the verified hardware is the caller's job.
    - RTMRs and TCB status are not appraised, and no Intel collateral (TCB
      info, QE identity collateral, CRLs) is evaluated. ``snp_report`` stays
      ``None``: there is no SNP report, and ``platform_integrity`` is
      SEV-SNP-only.
    """

    def __init__(
        self,
        trust_store: TrustStore,
        *,
        paravisor_measurements: Iterable[Union[str, bytes]],
    ) -> None:
        if isinstance(paravisor_measurements, (str, bytes)):
            raise TypeError("paravisor_measurements must be a collection, not one value")
        allowed = frozenset(_paravisor_measurement(m) for m in paravisor_measurements)
        if not allowed:
            raise ValueError("at least one Azure paravisor launch measurement is required")
        self._trust = trust_store
        self._paravisor_measurements = allowed

    def verify(
        self,
        quote_b64: str,
        *,
        expected_nonce: str,
        channel_binding: bytes = b"",
        expected_workload_measurement: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> QuoteVerification:
        # The bundle is attacker-supplied: a malformed bundle, HCL, runtime
        # JSON, ak_pem or TPM structure is a denial with a reason, never an
        # exception out of the release path. The DCAP quote itself is checked
        # by verify_tdx_quote, and anything that check raises is a denial too.
        try:
            bundle = unwrap_azure_tdx_vtpm_bundle(quote_b64)
        except ValueError as exc:
            return QuoteVerification(False, str(exc))
        try:
            ak = serialization.load_pem_public_key(bundle.ak_pem.encode())
        except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
            return QuoteVerification(False, f"invalid Azure vTPM evidence: {exc}")
        if not isinstance(ak, rsa.RSAPublicKey):
            return QuoteVerification(False, "HCL AK is not RSA")
        hcl = bundle.hcl
        if hcl[:4] != _HCL_MAGIC:
            return QuoteVerification(False, "not an Azure HCL report")
        if len(hcl) <= _HCL_REPORT_OFFSET or hcl[_HCL_REPORT_OFFSET] != _HCL_TD_REPORT_TYPE:
            return QuoteVerification(False, "HCL does not wrap a TDX TD report")

        try:
            parsed = parse_tdx_quote(bundle.dcap_quote)
        except QuoteFormatError as exc:
            return QuoteVerification(False, str(exc))
        current = now or datetime.now(timezone.utc)
        try:
            result = verify_tdx_quote(
                bundle.dcap_quote, self._trust, expected_nonce=None, now=current
            )
        except Exception:  # noqa: BLE001 - unverifiable evidence is a denial
            return QuoteVerification(False, "the DCAP quote could not be verified")
        if not result.verified:
            return result

        framing_error = _tdx_hcl_framing_error(hcl)
        if framing_error is not None:
            return QuoteVerification(False, framing_error)
        tdreport = hcl[_HCL_REPORT_OFFSET : _HCL_REPORT_OFFSET + _TDREPORT_LEN]
        mismatch = _tdreport_mismatch(tdreport, parsed.report.raw)
        if mismatch is not None:
            return QuoteVerification(
                False, f"HCL TD report does not match the Intel-signed TD report ({mismatch})"
            )
        if parsed.report.mrtd not in self._paravisor_measurements:
            return QuoteVerification(
                False, "TD MRTD is not an allowed Azure paravisor measurement"
            )
        if parsed.report.td_attributes & TD_ATTR_DEBUG:
            return QuoteVerification(False, "TDX TD attributes allow debugging")

        # The framing check above fixed where the runtime JSON is and how long.
        runtime_header = _HCL_REPORT_OFFSET + _HCL_SLOT_LEN
        json_size = _u32le(hcl, runtime_header + 16)
        json_start = runtime_header + _HCL_RUNTIME_HEADER_LEN
        runtime_json = hcl[json_start : json_start + json_size]
        # Bind the Intel-signed quote's REPORT_DATA, never the unsigned copy of
        # the TD report inside the HCL blob.
        if parsed.report.report_data[:32] != hashlib.sha256(runtime_json).digest():
            return QuoteVerification(False, "TDX REPORT_DATA does not bind HCL runtime JSON")
        if parsed.report.report_data[32:] != bytes(32):
            return QuoteVerification(False, "unexpected nonzero TDX REPORT_DATA tail")
        hcl_ak, ak_error = _tdx_hcl_authenticated_ak(runtime_json, ak)
        if hcl_ak is None:
            return QuoteVerification(False, ak_error)
        quote_error = _tdx_verify_pcr23_quote(
            bundle.tpm_quote,
            bundle.tpm_signature,
            hcl_ak,
            expected_nonce=expected_nonce,
            channel_binding=channel_binding,
            expected_workload_measurement=expected_workload_measurement,
        )
        if quote_error is not None:
            return QuoteVerification(False, quote_error)
        return QuoteVerification(True, leaf_subject=result.leaf_subject)
