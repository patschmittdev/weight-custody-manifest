"""Fail-closed verification of Azure SEV-SNP and Intel TDX plus fresh vTPM evidence.

Azure confidential VMs expose no ``/dev/sev-guest`` or ``/dev/tdx_guest``: the
paravisor publishes an HCL-wrapped hardware report in vTPM NV index 0x01400001
and binds that report's REPORT_DATA to the hash of a runtime-data JSON document
carrying the vTPM attestation key (``HCLAkPub``). The report therefore proves
the AK, not a caller nonce, so freshness has to come from a separate AK-signed
TPM quote over SHA-256 PCR 23 whose qualifying data is
``sha256(nonce || transport_key)``.

Both verifiers here check that chain fail-closed:

  ``AzureSnpVtpmVerifier``  AMD chain -> VCEK-signed SNP report -> runtime JSON
                            -> HCLAkPub -> AK-signed PCR 23 quote.
  ``AzureTdxVtpmVerifier``  Intel PCK chain -> DCAP-quoted TD report -> runtime
                            JSON -> HCLAkPub -> AK-signed PCR 23 quote.

The shared middle (runtime-data extraction, the HCLAkPub link and the PCR 23
quote check) is factored into the module-private helpers below so the two
platforms cannot drift apart.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from ._quote_verify import (
    QuoteFormatError,
    QuoteVerification,
    TrustStore,
    verify_cert_chain,
)
from ._certificates import load_pem_certificate
from .snp import extract_snp_report_from_hcl, parse_snp_report, verify_snp_report_signature
from .tdx import parse_tdx_quote, verify_tdx_quote

_HCL_MAGIC = b"HCLA"
_HCL_REPORT_OFFSET = 32
# The HCL report slot is 1184 bytes wide for SEV-SNP and for TDX alike, so the
# runtime-data region starts at the same offset on both platforms. Verified
# against a live Azure Standard_DC4es_v6 (TDX) capture as well as SEV-SNP hosts.
_HCL_RUNTIME_OFFSET = _HCL_REPORT_OFFSET + 1184
# REPORTMACSTRUCT TYPE byte of a TDX TD report, found at the HCL report offset.
# It is what distinguishes an Azure TDX CVM from an Azure SEV-SNP CVM sharing
# the same NV index.
_HCL_TD_REPORT_TYPE = 0x81

SNP_VTPM_BUNDLE_KIND = "wcm-azure-snp-vtpm/v1"
TDX_VTPM_BUNDLE_KIND = "wcm-azure-tdx-vtpm/v1"


@dataclass(frozen=True)
class AzureTdxVtpmBundle:
    """The decoded members of a ``wcm-azure-tdx-vtpm/v1`` bundle."""

    dcap_quote: bytes
    hcl: bytes
    ak_pem: str
    tpm_quote: bytes
    tpm_signature: bytes


def unwrap_azure_tdx_vtpm_bundle(quote_b64: str) -> AzureTdxVtpmBundle:
    """Decode the ``wcm-azure-tdx-vtpm/v1`` bundle carried in ``CpuQuote.quote_b64``.

    This is the one place that knows the bundle's wire shape. Consumers that
    need only the DCAP quote (MRTD extraction, ``wcm verify-quote``) call this
    rather than parsing the JSON themselves. Raises ``ValueError`` with a
    stable message on any shape problem: not base64 JSON, not an object, wrong
    ``kind``, or a missing or undecodable member.
    """
    try:
        doc = json.loads(_b64(quote_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Azure vTPM evidence: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("invalid Azure vTPM evidence: bundle is not a JSON object")
    kind = doc.get("kind")
    if kind != TDX_VTPM_BUNDLE_KIND:
        raise ValueError(f"unexpected Azure TDX vTPM bundle kind: {kind!r}")
    try:
        return AzureTdxVtpmBundle(
            dcap_quote=_b64(doc["tdx_quote_b64"]),
            hcl=_b64(doc["hcl_b64"]),
            ak_pem=str(doc["ak_pem"]),
            tpm_quote=_b64(doc["tpm_quote_b64"]),
            tpm_signature=_b64(doc["tpm_signature_b64"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid Azure vTPM evidence: {exc}") from exc


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


def _take_u16(blob: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 2 > len(blob):
        raise ValueError("truncated TPM2B")
    size = int.from_bytes(blob[offset : offset + 2], "big")
    start = offset + 2
    end = start + size
    if end > len(blob):
        raise ValueError("truncated TPM2B payload")
    return blob[start:end], end


def _hcl_runtime_json(hcl: bytes) -> tuple[Optional[bytes], Optional[str]]:
    """Return the HCL runtime-data JSON bytes, or ``(None, reason)``.

    The runtime region follows the report slot: a 20-byte header whose last four
    bytes are the little-endian JSON length, then the JSON document whose SHA-256
    the hardware report's REPORT_DATA binds.
    """
    runtime = hcl[_HCL_RUNTIME_OFFSET:]
    if len(runtime) < 20:
        return None, "HCL runtime data is truncated"
    json_len = int.from_bytes(runtime[16:20], "little")
    runtime_json = runtime[20 : 20 + json_len]
    if len(runtime_json) != json_len:
        return None, "HCL runtime JSON is truncated"
    return runtime_json, None


def _hcl_ak_modulus(runtime_json: bytes) -> tuple[Optional[int], Optional[str]]:
    """Return the ``HCLAkPub`` RSA modulus from the runtime JSON, or a reason."""
    try:
        runtime_doc = json.loads(runtime_json)
        jwk = next(k for k in runtime_doc["keys"] if k["kid"] == "HCLAkPub")
        modulus = int.from_bytes(base64.urlsafe_b64decode(jwk["n"] + "=" * (-len(jwk["n"]) % 4)), "big")
    except (KeyError, StopIteration, ValueError, TypeError) as exc:
        return None, f"invalid HCLAkPub: {exc}"
    return modulus, None


def _verify_pcr23_quote(
    quote: bytes,
    signature_blob: bytes,
    ak: rsa.RSAPublicKey,
    *,
    expected_nonce: str,
    channel_binding: bytes,
    expected_workload_measurement: Optional[str],
) -> Optional[str]:
    """Return a refusal reason for the AK-signed PCR 23 quote, or None if it holds.

    Checks, in order: the quote is TPM-generated and selects SHA-256 PCR 23; its
    signed ``pcrDigest`` is the measured-launch value the release policy approved;
    its qualifying data binds the challenge nonce and the transport key; and the
    AK's RSASSA-PKCS1v15/SHA-256 signature covers the quote verbatim.
    """
    try:
        if quote[:4] != b"\xffTCG" or quote[4:6] != b"\x80\x18":
            raise ValueError("not a TPM generated quote")
        _, offset = _take_u16(quote, 6)  # qualified signer
        extra_data, offset = _take_u16(quote, offset)
        offset += 25  # TPMS_CLOCK_INFO (17) + firmwareVersion (8)
        count = int.from_bytes(quote[offset : offset + 4], "big")
        offset += 4
        selected_pcr23 = False
        for _ in range(count):
            algorithm = int.from_bytes(quote[offset : offset + 2], "big")
            size = quote[offset + 2]
            selection = quote[offset + 3 : offset + 3 + size]
            offset += 3 + size
            selected_pcr23 |= algorithm == 0x000B and size >= 3 and bool(selection[2] & 0x80)
        if not selected_pcr23:
            return "TPM quote does not select SHA-256 PCR 23"
        pcr_digest, offset = _take_u16(quote, offset)
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


class AzureSnpVtpmVerifier:
    """Verify SNP authenticity, the HCL AK link, and a fresh AK-signed TPM quote.

    ``quote_b64`` is base64 JSON containing ``hcl_b64``, ``ak_pem``,
    ``tpm_quote_b64``, ``tpm_signature_b64``, ``vcek_pem``,
    ``intermediates_pem`` and ``root_pem``. The TPM quote must use
    SHA-256(nonce || channel_binding) as qualifying data and select PCR 23.
    """

    def __init__(self, trust_store: TrustStore) -> None:
        self._trust = trust_store

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
            doc: dict[str, Any] = json.loads(_b64(quote_b64))
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
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return QuoteVerification(False, f"invalid Azure vTPM evidence: {exc}")
        if not isinstance(ak, rsa.RSAPublicKey):
            return QuoteVerification(False, "HCL AK is not RSA")

        report = extract_snp_report_from_hcl(hcl)
        parsed = parse_snp_report(report)
        current = now or datetime.now(timezone.utc)
        chain_error = verify_cert_chain(vcek, intermediates, self._trust, current)
        if chain_error:
            return QuoteVerification(False, chain_error)
        if not verify_snp_report_signature(report, vcek):
            return QuoteVerification(False, "SNP report signature does not verify")

        runtime_json, runtime_error = _hcl_runtime_json(hcl)
        if runtime_json is None:
            return QuoteVerification(False, runtime_error)
        if parsed.report_data[:32] != hashlib.sha256(runtime_json).digest():
            return QuoteVerification(False, "SNP REPORT_DATA does not bind HCL runtime JSON")
        if parsed.report_data[32:] != bytes(32):
            return QuoteVerification(False, "unexpected nonzero SNP REPORT_DATA tail")
        modulus, ak_error = _hcl_ak_modulus(runtime_json)
        if modulus is None:
            return QuoteVerification(False, ak_error)
        if modulus != ak.public_numbers().n:
            return QuoteVerification(False, "TPM quote key does not match HCL-authenticated AK")

        quote_error = _verify_pcr23_quote(
            quote,
            signature_blob,
            ak,
            expected_nonce=expected_nonce,
            channel_binding=channel_binding,
            expected_workload_measurement=expected_workload_measurement,
        )
        if quote_error is not None:
            return QuoteVerification(False, quote_error)
        return QuoteVerification(True, leaf_subject=vcek.subject.rfc4514_string())


class AzureTdxVtpmVerifier:
    """Verify an Azure Intel TDX DCAP quote plus fresh AK-signed vTPM evidence.

    PROVISIONAL: not yet validated on hardware. The HCL layout, the runtime-data
    binding and the TPM quote format are taken from a captured Azure
    ``Standard_DC4es_v6`` TD report, but no end-to-end release has yet been run
    against a live Azure TDX CVM with this verifier.

    ``quote_b64`` is base64 JSON with ``kind`` ``wcm-azure-tdx-vtpm/v1`` plus
    ``tdx_quote_b64``, ``hcl_b64``, ``ak_pem``, ``tpm_quote_b64`` and
    ``tpm_signature_b64``. There is no certificate material at this level: the
    Intel PCK chain travels inside the DCAP quote.

    The TD report's REPORT_DATA is paravisor-bound rather than nonce-bound, so
    the DCAP quote is verified with ``expected_nonce=None`` and freshness is
    taken from the TPM quote's qualifying data,
    ``sha256(nonce || channel_binding)``, over SHA-256 PCR 23.
    """

    def __init__(self, trust_store: TrustStore) -> None:
        self._trust = trust_store

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
            bundle = unwrap_azure_tdx_vtpm_bundle(quote_b64)
            ak = serialization.load_pem_public_key(bundle.ak_pem.encode())
        except (ValueError, TypeError) as exc:
            reason = str(exc)
            if not reason.startswith(("invalid Azure vTPM evidence", "unexpected Azure TDX vTPM bundle kind")):
                reason = f"invalid Azure vTPM evidence: {exc}"
            return QuoteVerification(False, reason)
        dcap = bundle.dcap_quote
        hcl = bundle.hcl
        quote = bundle.tpm_quote
        signature_blob = bundle.tpm_signature
        if not isinstance(ak, rsa.RSAPublicKey):
            return QuoteVerification(False, "HCL AK is not RSA")
        if hcl[:4] != _HCL_MAGIC:
            return QuoteVerification(False, "not an Azure HCL report")
        if len(hcl) <= _HCL_REPORT_OFFSET or hcl[_HCL_REPORT_OFFSET] != _HCL_TD_REPORT_TYPE:
            return QuoteVerification(False, "HCL does not wrap a TDX TD report")

        try:
            parsed = parse_tdx_quote(dcap)
        except QuoteFormatError as exc:
            return QuoteVerification(False, str(exc))
        current = now or datetime.now(timezone.utc)
        result = verify_tdx_quote(dcap, self._trust, expected_nonce=None, now=current)
        if not result.verified:
            return result

        runtime_json, runtime_error = _hcl_runtime_json(hcl)
        if runtime_json is None:
            return QuoteVerification(False, runtime_error)
        # Bind the Intel-signed quote's REPORT_DATA, never the unsigned copy of
        # the TD report inside the HCL blob: only the former is covered by the
        # attestation key's signature.
        if parsed.report.report_data[:32] != hashlib.sha256(runtime_json).digest():
            return QuoteVerification(False, "TDX REPORT_DATA does not bind HCL runtime JSON")
        if parsed.report.report_data[32:] != bytes(32):
            return QuoteVerification(False, "unexpected nonzero TDX REPORT_DATA tail")
        modulus, ak_error = _hcl_ak_modulus(runtime_json)
        if modulus is None:
            return QuoteVerification(False, ak_error)
        if modulus != ak.public_numbers().n:
            return QuoteVerification(False, "TPM quote key does not match HCL-authenticated AK")

        quote_error = _verify_pcr23_quote(
            quote,
            signature_blob,
            ak,
            expected_nonce=expected_nonce,
            channel_binding=channel_binding,
            expected_workload_measurement=expected_workload_measurement,
        )
        if quote_error is not None:
            return QuoteVerification(False, quote_error)
        return QuoteVerification(True, leaf_subject=parsed.pck_leaf.subject.rfc4514_string())
