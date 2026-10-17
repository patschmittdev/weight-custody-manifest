"""AzureTdxVtpmProvider: what's testable without an Azure TDX CVM.

The vTPM read and the IMDS /acc/tdquote call only run on-guest; CI covers
availability detection, fail-closed behavior off-guest, the default raw-DCAP
path, opt-in bundle plumbing, measured-launch ordering inside the bundle, and
the shape of the tpm2 commands the shared freshness helper issues.
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


def test_availability_keeps_the_raw_dcap_tool_requirements(monkeypatch):
    requested: list[str] = []

    monkeypatch.setattr("wcm._hw_providers.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "wcm._hw_providers.shutil.which",
        lambda name: requested.append(name) or "/usr/bin/" + name,
    )
    monkeypatch.setattr(AzureTdxVtpmProvider, "_fetch_hcl", lambda self: _synth_hcl_tdx())

    assert AzureTdxVtpmProvider.is_available() is True
    assert requested == ["tpm2_nvread"]


def test_cpu_quote_raises_off_guest():
    p = AzureTdxVtpmProvider()  # real _fetch_hcl -> no tpm2_nvread
    with pytest.raises(AttestationUnavailableError):
        p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "0" * 64)


def test_default_output_is_raw_dcap_without_pcr_or_ak_quote_calls():
    p = AzureTdxVtpmProvider()
    hcl = _synth_hcl_tdx()
    captured = {}

    def fake_quote(tdreport: bytes) -> bytes:
        captured["tdreport"] = tdreport
        return b"FAKE-DCAP-TD-QUOTE-BYTES"

    def fail_measurement(measurement: str) -> None:
        pytest.fail("default path mutated PCR 23")

    def fail_ak_quote(binding: bytes) -> tuple[str, bytes, bytes]:
        pytest.fail("default path took an AK quote")

    p._fetch_hcl = lambda: hcl  # type: ignore[method-assign]
    p._fetch_quote = fake_quote  # type: ignore[method-assign]
    p._measure_workload = fail_measurement  # type: ignore[method-assign]
    p._tpm_pcr23_quote = fail_ak_quote  # type: ignore[method-assign]

    ch = _challenge()
    quote = p.cpu_quote(ch, serving_image_measurement="sha256:" + "ab" * 32)
    assert quote.platform == "intel-tdx"
    assert quote.nonce_echo == ch.nonce
    assert quote.attestation_key_id == "tdx-quote:azure-vtpm"
    # The TD report handed to the quote service is the 1024 bytes at HCL offset 32.
    assert len(captured["tdreport"]) == 1024
    assert captured["tdreport"][0] == 0x81

    assert base64.b64decode(quote.quote_b64) == b"FAKE-DCAP-TD-QUOTE-BYTES"


def test_opt_in_output_is_vtpm_bundle():
    p = AzureTdxVtpmProvider(vtpm_bundle=True)
    hcl = _synth_hcl_tdx()
    p._fetch_hcl = lambda: hcl  # type: ignore[method-assign]
    p._fetch_quote = lambda tdreport: b"FAKE-DCAP-TD-QUOTE-BYTES"  # type: ignore[method-assign]
    p._measure_workload = lambda measurement: None  # type: ignore[method-assign]
    p._tpm_pcr23_quote = lambda binding: (  # type: ignore[method-assign]
        "-----BEGIN PUBLIC KEY-----fake-----END PUBLIC KEY-----",
        b"FAKE-TPMS-ATTEST",
        b"FAKE-TPMT-SIGNATURE",
    )

    quote = p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "ab" * 32)
    bundle = json.loads(base64.b64decode(quote.quote_b64))
    assert bundle["kind"] == "wcm-azure-tdx-vtpm/v1"
    assert base64.b64decode(bundle["tdx_quote_b64"]) == b"FAKE-DCAP-TD-QUOTE-BYTES"
    assert base64.b64decode(bundle["hcl_b64"]) == hcl
    assert "BEGIN PUBLIC KEY" in bundle["ak_pem"]
    assert base64.b64decode(bundle["tpm_quote_b64"]) == b"FAKE-TPMS-ATTEST"
    assert base64.b64decode(bundle["tpm_signature_b64"]) == b"FAKE-TPMT-SIGNATURE"


def test_cpu_quote_measures_before_quoting():
    """PCR 23 is extended before the AK quotes it, or the quote proves nothing."""
    p = AzureTdxVtpmProvider(vtpm_bundle=True)
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
    """A failed measured launch prevents the AK quote and aborts evidence production."""
    p = AzureTdxVtpmProvider(vtpm_bundle=True)
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
    p = AzureTdxVtpmProvider(vtpm_bundle=True)
    body = bytearray(_synth_hcl_tdx())
    body[32] = 0x00
    p._fetch_hcl = lambda: bytes(body)  # type: ignore[method-assign]
    p._fetch_quote = lambda tdreport: b"FAKE-DCAP"  # type: ignore[method-assign]
    p._measure_workload = lambda measurement: None  # type: ignore[method-assign]

    with pytest.raises(AttestationUnavailableError, match="does not hold a TDX HCL report"):
        p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "ab" * 32)


def test_cpu_quote_rejects_truncated_td_report_before_quoting():
    """An HCL whose TD report is cut short never reaches /acc/tdquote or PCR 23."""
    p = AzureTdxVtpmProvider(vtpm_bundle=True)
    hcl = _synth_hcl_tdx()[: 32 + 1023]
    assert hcl[32] == 0x81
    calls: list[str] = []
    p._fetch_hcl = lambda: hcl  # type: ignore[method-assign]
    p._fetch_quote = lambda tdreport: calls.append("quote") or b"FAKE-DCAP"  # type: ignore[method-assign]
    p._measure_workload = lambda measurement: calls.append("measure")  # type: ignore[method-assign]

    with pytest.raises(AttestationUnavailableError, match="does not hold a TDX HCL report"):
        p.cpu_quote(_challenge(), serving_image_measurement="sha256:" + "ab" * 32)
    assert calls == []


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
    p = AzureTdxVtpmProvider(vtpm_bundle=True)
    ak_pem, quote, signature = p._tpm_pcr23_quote(binding)
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
    p = AzureTdxVtpmProvider(vtpm_bundle=True)
    with pytest.raises(AttestationUnavailableError, match="tpm2_readpublic not found"):
        p._tpm_pcr23_quote(bytes(32))


def _fake_hardware_provider():
    """An AzureTdxVtpmProvider whose hardware calls are stubbed, and its trust root.

    The fake quote service signs a TD report body derived from the TDREPORT the
    provider actually hands it, so the HCL and the DCAP quote agree the way
    they do on Azure. PCR 23 holds the digest the provider measured, and the
    AK quote is built from the qualifying data the provider hands to
    tpm2_quote.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    from tests.test_azure_tdx_vtpm_verify import (
        _hcl,
        _pem,
        _runtime_json,
        _tdreport,
        _tpm2b,
        dcap_body,
    )
    from tests.test_tdx import build_quote
    from wcm import TrustStore
    from wcm.azure_vtpm import expected_pcr23_digest

    ak_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    runtime_json = _runtime_json(ak_key.public_key().public_numbers().n)
    hcl = _hcl(runtime_json, _tdreport(runtime_json))
    trust = TrustStore()
    measured: list[str] = []

    def fake_quote(tdreport: bytes) -> bytes:
        assert tdreport == hcl[32 : 32 + 1024]
        dcap, root = build_quote(td_body=dcap_body(tdreport))
        trust.add_root(root)
        return dcap

    def fake_tpm(binding: bytes) -> tuple[str, bytes, bytes]:
        quote = (
            b"\xffTCG\x80\x18" + _tpm2b(b"signer") + _tpm2b(binding) + bytes(25)
            + (1).to_bytes(4, "big") + b"\x00\x0b\x03\x00\x00\x80"
            + _tpm2b(expected_pcr23_digest(measured[-1]))
        )
        signature = ak_key.sign(quote, padding.PKCS1v15(), hashes.SHA256())
        return _pem(ak_key.public_key()), quote, b"\x00\x14\x00\x0b" + _tpm2b(signature)

    p = AzureTdxVtpmProvider(vtpm_bundle=True)
    p._fetch_hcl = lambda: hcl  # type: ignore[method-assign]
    p._fetch_quote = fake_quote  # type: ignore[method-assign]
    p._measure_workload = measured.append  # type: ignore[method-assign]
    p._tpm_pcr23_quote = fake_tpm  # type: ignore[method-assign]
    return p, trust


def test_provider_bundle_verifies_with_the_kbs_channel_binding():
    """Provider and verifier agree on sha256(nonce || transport key) and PCR 23."""
    from tests.test_azure_tdx_vtpm_verify import PARAVISOR
    from tests.test_tdx import NOW
    from wcm import AzureTdxVtpmVerifier

    p, trust = _fake_hardware_provider()
    measurement = "sha256:" + "ab" * 32
    transport_key = "cd" * 32
    ch = _challenge()
    cpu = p.cpu_quote(ch, serving_image_measurement=measurement, transport_public_key=transport_key)
    verifier = AzureTdxVtpmVerifier(trust, paravisor_measurements=[PARAVISOR])
    result = verifier.verify(
        cpu.quote_b64,
        expected_nonce=ch.nonce,
        channel_binding=bytes.fromhex(transport_key),
        expected_workload_measurement=measurement,
        now=NOW,
    )
    assert result.verified, result.reason
    assert result.snp_report is None
    other = verifier.verify(
        cpu.quote_b64,
        expected_nonce=ch.nonce,
        channel_binding=bytes.fromhex("ee" * 32),
        expected_workload_measurement=measurement,
        now=NOW,
    )
    assert other.reason == "TPM qualifying data does not bind nonce and transport key"


def test_raw_dcap_consumers_must_unwrap_the_bundle_first():
    """The opt-in bundle is not raw DCAP: raw-DCAP consumers must unwrap it first."""
    from tests.test_tdx import NOW
    from wcm import unwrap_azure_tdx_vtpm_bundle
    from wcm._quote_verify import QuoteFormatError
    from wcm.platform_floor import PlatformFloor, appraise_tdx_quote
    from wcm.tdx import parse_tdx_quote, verify_tdx_quote

    p, trust = _fake_hardware_provider()
    cpu = p.cpu_quote(
        _challenge(), serving_image_measurement="sha256:" + "ab" * 32, transport_public_key="cd" * 32
    )
    bundle_bytes = base64.b64decode(cpu.quote_b64)
    with pytest.raises(QuoteFormatError):
        parse_tdx_quote(bundle_bytes)
    assert not verify_tdx_quote(bundle_bytes, trust, expected_nonce=None, now=NOW).verified
    appraisal = appraise_tdx_quote(bundle_bytes, PlatformFloor())
    assert [c.name for c in appraisal.checks] == ["intel-tdx.quote"]
    assert appraisal.checks[0].state.value == "failed"
    assert "does not parse" in (appraisal.checks[0].reason or "")
    dcap = unwrap_azure_tdx_vtpm_bundle(cpu.quote_b64).dcap_quote
    assert verify_tdx_quote(dcap, trust, expected_nonce=None, now=NOW).verified


def test_raw_default_passes_the_raw_verifier_and_is_refused_by_the_bundle_verifier():
    """The default raw DCAP quote verifies as DCAP only; the full-bundle verifier refuses it."""
    from tests.test_azure_tdx_vtpm_verify import PARAVISOR
    from tests.test_tdx import NOW
    from wcm import AzureTdxVtpmVerifier
    from wcm.tdx import verify_tdx_quote

    p, trust = _fake_hardware_provider()
    p._vtpm_bundle = False
    ch = _challenge()
    cpu = p.cpu_quote(ch, serving_image_measurement="sha256:" + "ab" * 32, transport_public_key="cd" * 32)
    assert verify_tdx_quote(base64.b64decode(cpu.quote_b64), trust, expected_nonce=None, now=NOW).verified
    result = AzureTdxVtpmVerifier(trust, paravisor_measurements=[PARAVISOR]).verify(
        cpu.quote_b64,
        expected_nonce=ch.nonce,
        channel_binding=bytes.fromhex("cd" * 32),
        expected_workload_measurement="sha256:" + "ab" * 32,
        now=NOW,
    )
    assert not result.verified
    assert (result.reason or "").startswith("invalid Azure vTPM evidence: ")


def _kbs_release(example_dict, *, platform_integrity=None):
    """Run a TDX bundle from the provider through KeyBrokerService end to end."""
    import copy

    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from tests.conftest import waive
    from tests.test_azure_tdx_vtpm_verify import PARAVISOR
    from tests.test_tdx import NOW
    from wcm import (
        AzureTdxVtpmVerifier,
        KeyBrokerService,
        SoftwareProvider,
        WeightCustodyManifest,
        manifest_identity,
    )

    doc = copy.deepcopy(example_dict)
    doc["release_policy"]["required_hw_platform"] = ["intel-tdx", "nvidia-cc-gpu"]
    if platform_integrity is not None:
        doc["release_policy"]["platform_integrity"] = platform_integrity
    # The manifest waives evidence verification only so the mock GPU report
    # (no GPU verifier here) can pass; the CPU quote verifier is configured
    # and required, so the TDX bundle is verified cryptographically.
    manifest = waive(WeightCustodyManifest.model_validate(doc))
    policy = manifest.release_policy
    current = next(
        m.measurement
        for m in policy.required_serving_image.accepted_measurements
        if m.status.value == "current"
    )
    provider, trust = _fake_hardware_provider()
    kbs = KeyBrokerService(
        {manifest.weights_hash: b"decryption-key-for-these-weights"},
        now=lambda: NOW,
        trusted_manifest_identities={manifest_identity(manifest)},
        cpu_quote_verifier=AzureTdxVtpmVerifier(trust, paravisor_measurements=[PARAVISOR]),
        require_cpu_quote_verification=True,
        require_channel_binding=True,
    )
    transport_key = X25519PrivateKey.generate().public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    ).hex()
    challenge = kbs.issue_challenge()
    evidence = SoftwareProvider().produce(
        challenge,
        serving_image_measurement=str(current),
        platform="intel-tdx",
        gpu_measurement=policy.required_gpu_measurement.rim_pin,
        transport_public_key=transport_key,
    )
    cpu = provider.cpu_quote(
        challenge, serving_image_measurement=str(current), transport_public_key=transport_key
    )
    evidence = evidence.model_copy(update={"cpu": cpu})
    return kbs.verify_and_release(manifest, evidence)


def _check(decision, name: str):
    return next(c for c in decision.checks if c.name == name)


def test_tdx_bundle_releases_through_the_key_broker(example_dict):
    decision = _kbs_release(example_dict)
    cpu_check = _check(decision, "cpu_quote_verified")
    assert cpu_check.passed, cpu_check
    assert "waived" not in (cpu_check.detail or "")
    assert decision.released, [c for c in decision.checks if not c.passed]


def test_platform_integrity_stays_sev_snp_only_for_tdx(example_dict):
    decision = _kbs_release(example_dict, platform_integrity={"alias_check_complete": "required"})
    check = _check(decision, "platform_integrity")
    assert not check.passed
    assert "SEV-SNP-only" in check.detail
    assert _check(decision, "cpu_quote_verified").passed
    assert not decision.released
