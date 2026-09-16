"""AzureTdxVtpmProvider: what's testable without an Azure TDX CVM.

The vTPM read and the IMDS /acc/tdquote call only run on-guest; CI covers
availability detection, fail-closed behavior off-guest, the
HCL -> TD report -> quote plumbing via mocked fetches, the measured-launch
ordering inside the evidence bundle, and the shape of the tpm2 commands the
shared freshness helper issues.
"""
from __future__ import annotations

import base64
import json

import pytest

from wcm import AttestationUnavailableError, AzureTdxVtpmProvider, ChallengeStore


def _challenge():
    return ChallengeStore().issue()


def _synth_hcl_tdx() -> bytes:
    # HCLA header, then a TD report at offset 32 whose REPORTMACSTRUCT TYPE byte
    # is 0x81 (TDX) - the marker AzureTdxVtpmProvider keys on.
    body = bytearray(2600)
    body[:4] = b"HCLA"
    body[32] = 0x81
    body[33:64] = bytes(range(31))  # arbitrary TD report bytes
    return bytes(body)


def test_unavailable_off_guest():
    # No /dev/tpmrm0 on CI/Windows, so the Azure TDX provider is not available.
    assert AzureTdxVtpmProvider.is_available() is False


def test_cpu_quote_raises_off_guest():
    p = AzureTdxVtpmProvider()  # real _fetch_hcl -> no tpm2_nvread
    with pytest.raises(AttestationUnavailableError):
        p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "0" * 64)


def test_plumbing_with_mocked_fetches():
    p = AzureTdxVtpmProvider()
    hcl = _synth_hcl_tdx()
    captured = {}

    def fake_quote(tdreport: bytes) -> bytes:
        captured["tdreport"] = tdreport
        return b"FAKE-DCAP-TD-QUOTE-BYTES"

    p._fetch_hcl = lambda: hcl  # type: ignore[method-assign]
    p._fetch_quote = fake_quote  # type: ignore[method-assign]
    p._measure_workload = lambda measurement: None  # type: ignore[method-assign]
    p._tpm_pcr23_quote = lambda binding: (  # type: ignore[method-assign]
        "-----BEGIN PUBLIC KEY-----fake-----END PUBLIC KEY-----",
        b"FAKE-TPMS-ATTEST",
        b"FAKE-TPMT-SIGNATURE",
    )

    ch = _challenge()
    quote = p.cpu_quote(ch, serving_image_measurement="sha256:" + "ab" * 32)
    assert quote.platform == "intel-tdx"
    assert quote.nonce_echo == ch.nonce
    assert quote.attestation_key_id == "tdx-quote:azure-vtpm"
    # The TD report handed to the quote service is the 1024 bytes at HCL offset 32.
    assert len(captured["tdreport"]) == 1024
    assert captured["tdreport"][0] == 0x81

    bundle = json.loads(base64.b64decode(quote.quote_b64))
    assert bundle["kind"] == "wcm-azure-tdx-vtpm/v1"
    assert base64.b64decode(bundle["tdx_quote_b64"]) == b"FAKE-DCAP-TD-QUOTE-BYTES"
    assert base64.b64decode(bundle["hcl_b64"]) == hcl
    assert "BEGIN PUBLIC KEY" in bundle["ak_pem"]
    assert base64.b64decode(bundle["tpm_quote_b64"]) == b"FAKE-TPMS-ATTEST"
    assert base64.b64decode(bundle["tpm_signature_b64"]) == b"FAKE-TPMT-SIGNATURE"


def test_cpu_quote_measures_before_quoting():
    """PCR 23 is extended before the AK quotes it, or the quote proves nothing."""
    p = AzureTdxVtpmProvider()
    order: list[str] = []

    p._fetch_hcl = lambda: _synth_hcl_tdx()  # type: ignore[method-assign]

    def fake_fetch_quote(tdreport: bytes) -> bytes:
        order.append("fetch_quote")
        return b"FAKE-DCAP"

    def fake_measure(measurement: str) -> None:
        order.append("measure")

    def fake_tpm(binding: bytes) -> tuple[str, bytes, bytes]:
        order.append("tpm_quote")
        return "ak", b"m", b"s"

    p._fetch_quote = fake_fetch_quote  # type: ignore[method-assign]
    p._measure_workload = fake_measure  # type: ignore[method-assign]
    p._tpm_pcr23_quote = fake_tpm  # type: ignore[method-assign]

    p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "ab" * 32)
    assert order.index("measure") < order.index("tpm_quote")
    assert order.index("fetch_quote") < order.index("measure")


def test_cpu_quote_fails_closed_when_measure_fails():
    """A failed measured launch aborts evidence production; nothing gets quoted."""
    p = AzureTdxVtpmProvider()
    quoted: list[bytes] = []

    def refuse(measurement: str) -> None:
        raise AttestationUnavailableError("PCR 23 reset/extend failed")

    def record(binding: bytes) -> tuple[str, bytes, bytes]:
        quoted.append(binding)
        return "ak", b"m", b"s"

    p._fetch_hcl = lambda: _synth_hcl_tdx()  # type: ignore[method-assign]
    p._fetch_quote = lambda tdreport: b"FAKE-DCAP"  # type: ignore[method-assign]
    p._measure_workload = refuse  # type: ignore[method-assign]
    p._tpm_pcr23_quote = record  # type: ignore[method-assign]

    with pytest.raises(AttestationUnavailableError):
        p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "ab" * 32)
    assert quoted == []


def test_cpu_quote_rejects_non_tdx_hcl():
    """An SNP HCL (TYPE byte 0x00) in the same NV index is refused, not quoted."""
    p = AzureTdxVtpmProvider()
    body = bytearray(_synth_hcl_tdx())
    body[32] = 0x00
    p._fetch_hcl = lambda: bytes(body)  # type: ignore[method-assign]
    p._fetch_quote = lambda tdreport: b"FAKE-DCAP"  # type: ignore[method-assign]
    p._measure_workload = lambda measurement: None  # type: ignore[method-assign]

    with pytest.raises(AttestationUnavailableError, match="does not hold a TDX HCL report"):
        p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "ab" * 32)


def test_tpm_pcr23_quote_command_shape(monkeypatch):
    """The shared freshness helper quotes SHA-256 PCR 23 over the binding."""
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        flag = "-o" if argv[0] == "tpm2_readpublic" else None
        if flag is not None:
            with open(argv[argv.index(flag) + 1], "w", encoding="utf-8") as fh:
                fh.write("-----BEGIN PUBLIC KEY-----fake-----END PUBLIC KEY-----")
        else:
            for out, payload in (("-m", b"TPMS-ATTEST"), ("-s", b"TPMT-SIGNATURE")):
                with open(argv[argv.index(out) + 1], "wb") as fh:
                    fh.write(payload)
        return None

    monkeypatch.setattr("wcm._hw_providers.shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("wcm._hw_providers.subprocess.run", fake_run)

    binding = bytes(range(32))
    ak_pem, quote, signature = AzureTdxVtpmProvider()._tpm_pcr23_quote(binding)
    assert "BEGIN PUBLIC KEY" in ak_pem
    assert quote == b"TPMS-ATTEST" and signature == b"TPMT-SIGNATURE"

    readpublic, tpm_quote = calls
    assert readpublic[:4] == ["tpm2_readpublic", "-c", "0x81000003", "-f"]
    assert tpm_quote[:2] == ["tpm2_quote", "-c"]
    assert tpm_quote[tpm_quote.index("-l") + 1] == "sha256:23"
    assert tpm_quote[tpm_quote.index("-q") + 1] == binding.hex()
    assert tpm_quote[tpm_quote.index("-g") + 1] == "sha256"


def test_tpm_pcr23_quote_requires_tooling(monkeypatch):
    monkeypatch.setattr("wcm._hw_providers.shutil.which", lambda name: None)
    with pytest.raises(AttestationUnavailableError, match="tpm2_readpublic not found"):
        AzureTdxVtpmProvider()._tpm_pcr23_quote(bytes(32))
