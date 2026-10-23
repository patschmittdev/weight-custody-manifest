"""CLI coverage for the verify/gate/inspect verbs, driven through ``main()``.

verify-quote is exercised against the committed fixtures for all three kinds
(SNP, TDX, GPU), happy path plus wrong-nonce and untrusted-root failures. gate
and inspect run against the example manifest. verify-provenance is guarded so it
skips when the model-signing extra is absent and runs for real when it is.
"""
from __future__ import annotations

import copy
import json
import pathlib
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from wcm._vendor_vectors import roots_dir
from wcm.cli import main

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BAD_NONCE = "cd" * 32

try:
    import model_signing  # noqa: F401

    HAVE_MS = True
except ImportError:
    HAVE_MS = False

requires_ms = pytest.mark.skipif(not HAVE_MS, reason="model-signing extra not installed")


def _stranger_root(tmp_path: pathlib.Path) -> str:
    key = ec.generate_private_key(ec.SECP384R1())
    now = datetime.now(timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "stranger-root")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    p = tmp_path / "stranger.pem"
    p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(p)


def _manifest_file(tmp_path: pathlib.Path, doc: dict) -> str:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


# -- verify-quote: SNP ---------------------------------------------------------


def _synthetic_snp_root(tmp_path: pathlib.Path) -> str:
    # Trusting the synthetic ARK has to be an explicit caller decision.
    bundle = json.loads((FIXTURES / "snp_quote_synthetic.json").read_text())
    p = tmp_path / "synthetic-ark.pem"
    p.write_text(bundle["root_pem"], encoding="utf-8")
    return str(p)


def test_vq_snp_synthetic_ok(capsys, tmp_path):
    rc = main(
        ["verify-quote", "--kind", "snp", str(FIXTURES / "snp_quote_synthetic.json"),
         "--root", _synthetic_snp_root(tmp_path)]
    )
    assert rc == 0
    assert "verified  : True" in capsys.readouterr().out


def test_vq_snp_bundled_root_is_not_trusted_unless_pinned(capsys):
    # The bundle is the evidence under test; the root it carries used to be
    # trusted as is, so a fully self-made chain verified with exit code 0.
    rc = main(["verify-quote", "--kind", "snp", str(FIXTURES / "snp_quote_synthetic.json")])
    assert rc == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out and "not a pinned AMD ARK" in out


def test_vq_snp_wrong_nonce(capsys, tmp_path):
    rc = main(
        ["verify-quote", "--kind", "snp", str(FIXTURES / "snp_quote_synthetic.json"),
         "--nonce", BAD_NONCE, "--root", _synthetic_snp_root(tmp_path)]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out and "REPORT_DATA" in out


def test_vq_snp_untrusted_root(capsys, tmp_path):
    rc = main(
        [
            "verify-quote",
            "--kind",
            "snp",
            str(FIXTURES / "snp_quote_synthetic.json"),
            "--root",
            _stranger_root(tmp_path),
        ]
    )
    assert rc == 1
    assert "trusted root" in capsys.readouterr().out.lower()


def test_vq_snp_azure_vtpm_rejects_nonconforming_certificate(capsys):
    # This sanitized genuine capture has a non-positive provider-certificate
    # serial. RFC 5280 disallows it and cryptography 51 refuses to parse it, so
    # WCM gives the same fail-closed verdict on cryptography 50.
    assert main(
        [
            "verify-quote",
            "--kind",
            "snp",
            str(FIXTURES / "snp_quote_azure.json"),
            "--root",
            str(roots_dir() / "amd-ark-milan.pem"),
        ]
    ) == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out
    assert "serial number must be positive" in out


# -- verify-quote: TDX ---------------------------------------------------------


def test_vq_tdx_ok(capsys):
    assert main(["verify-quote", "--kind", "tdx", str(FIXTURES / "tdx_quote_gcp.json")]) == 0
    assert "verified  : True" in capsys.readouterr().out


def test_vq_tdx_wrong_nonce(capsys):
    rc = main(
        ["verify-quote", "--kind", "tdx", str(FIXTURES / "tdx_quote_gcp.json"), "--nonce", BAD_NONCE]
    )
    assert rc == 1
    assert "verified  : False" in capsys.readouterr().out


def test_vq_tdx_bundle_cannot_supply_its_own_root_pin(capsys, tmp_path):
    import base64
    import hashlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_tdx import NONCE, build_quote

    quote, root = build_quote()
    bundle = {
        "quote_b64": base64.b64encode(quote).decode(),
        "expected_nonce": NONCE,
        "intel_sgx_root_ca_sha256": hashlib.sha256(
            root.public_bytes(serialization.Encoding.DER)
        ).hexdigest(),
    }
    path = tmp_path / "self-pinned.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    assert main(["verify-quote", "--kind", "tdx", str(path)]) == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out and "pinned Intel SGX root" in out


def test_vq_tdx_untrusted_root(capsys, tmp_path):
    rc = main(
        [
            "verify-quote",
            "--kind",
            "tdx",
            str(FIXTURES / "tdx_quote_gcp.json"),
            "--root",
            _stranger_root(tmp_path),
        ]
    )
    assert rc == 1
    assert "verified  : False" in capsys.readouterr().out


def test_vq_tdx_garbage_quote_is_a_denial(capsys, tmp_path):
    # parse_tdx_quote raises QuoteFormatError, which is not a ValueError, so
    # this used to escape cmd_verify_quote as a traceback.
    import base64

    path = tmp_path / "garbage.json"
    path.write_text(json.dumps({"quote_b64": base64.b64encode(b"\0" * 3).decode()}))
    assert main(["verify-quote", "--kind", "tdx", str(path)]) == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out and "TDX quote too short" in out


def test_vq_missing_input_reports_one_line_error(capsys, tmp_path):
    missing = tmp_path / "missing.json"
    assert main(["verify-quote", "--kind", "tdx", str(missing)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(f"error: {missing}: ")
    assert output.err.count("\n") == 1
    assert "Traceback" not in output.err


def test_vq_invalid_json_input_reports_one_line_error(capsys, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")
    assert main(["verify-quote", "--kind", "snp", str(path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(f"error: {path}: ")
    assert output.err.count("\n") == 1
    assert "Traceback" not in output.err


def test_vq_missing_root_file_reports_one_line_error(capsys, tmp_path):
    path = tmp_path / "quote.json"
    path.write_text(json.dumps({"quote_b64": "AAAA"}), encoding="utf-8")
    missing = tmp_path / "missing-root.pem"
    assert main(["verify-quote", "--kind", "tdx", str(path), "--root", str(missing)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(f"error: {missing}: ")
    assert output.err.count("\n") == 1
    assert "Traceback" not in output.err


def _azure_vtpm_bundle(
    tmp_path: pathlib.Path, *, kind: str, dcap_b64: str, outer: dict | None = None
) -> str:
    """Wrap a DCAP quote in a wcm-azure-tdx-vtpm/v1 bundle on disk."""
    import base64

    inner = {
        "kind": kind,
        "tdx_quote_b64": dcap_b64,
        # Placeholders: this verb checks the DCAP half only.
        "hcl_b64": base64.b64encode(b"HCLA-placeholder").decode(),
        "ak_pem": "-----BEGIN PUBLIC KEY-----placeholder-----END PUBLIC KEY-----",
        "tpm_quote_b64": base64.b64encode(b"placeholder").decode(),
        "tpm_signature_b64": base64.b64encode(b"placeholder").decode(),
    }
    doc = {"quote_b64": base64.b64encode(json.dumps(inner).encode()).decode()}
    doc.update(outer or {})
    out = tmp_path / "azure-tdx-vtpm.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    return str(out)


def _azure_dcap_b64() -> str:
    return json.loads((FIXTURES / "tdx_quote_azure.json").read_text(encoding="utf-8"))["quote_b64"]


def test_vq_tdx_azure_vtpm_bundle_gets_no_verdict_without_dcap_only(capsys, tmp_path):
    # Only the DCAP half can be checked here, so a bundle never earns an
    # overall success verdict by default.
    path = _azure_vtpm_bundle(tmp_path, kind="wcm-azure-tdx-vtpm/v1", dcap_b64=_azure_dcap_b64())
    assert main(["verify-quote", "--kind", "tdx", path]) == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out
    assert "gives no verdict on the bundle" in out and "--dcap-only" in out
    assert "scope" not in out


def test_vq_tdx_azure_vtpm_bundle_dcap_only_is_scoped(capsys, tmp_path):
    path = _azure_vtpm_bundle(tmp_path, kind="wcm-azure-tdx-vtpm/v1", dcap_b64=_azure_dcap_b64())
    assert main(["verify-quote", "--kind", "tdx", path, "--dcap-only"]) == 0
    out = capsys.readouterr().out
    assert "scope     : dcap-only" in out
    assert "verified  : True" in out
    assert "inner DCAP quote only" in out and "not checked" in out


@pytest.mark.parametrize("member", ["kind", "tdx_quote_b64"])
def test_vq_tdx_azure_vtpm_duplicate_bundle_member_is_a_denial(capsys, tmp_path, member):
    import base64

    path = _azure_vtpm_bundle(tmp_path, kind="wcm-azure-tdx-vtpm/v1", dcap_b64=_azure_dcap_b64())
    args = ["verify-quote", "--kind", "tdx", path, "--dcap-only"]
    assert main(args) == 0
    assert "verified  : True" in capsys.readouterr().out
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    raw = base64.b64decode(doc["quote_b64"]).decode()
    duplicate = json.dumps({member: "wrong"})[:-1] + "," + raw[1:]
    doc["quote_b64"] = base64.b64encode(duplicate.encode()).decode()
    pathlib.Path(path).write_text(json.dumps(doc), encoding="utf-8")
    assert main(args) == 1
    output = capsys.readouterr()
    assert "verified  : False" in output.out
    assert f"duplicate JSON object key {member!r}" in output.out
    assert "Traceback" not in output.out + output.err


def test_vq_tdx_azure_vtpm_bundle_with_leading_whitespace_is_recognised(capsys, tmp_path):
    import base64

    path = _azure_vtpm_bundle(tmp_path, kind="wcm-azure-tdx-vtpm/v1", dcap_b64=_azure_dcap_b64())
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    doc["quote_b64"] = base64.b64encode(b" \n\t" + base64.b64decode(doc["quote_b64"])).decode()
    pathlib.Path(path).write_text(json.dumps(doc), encoding="utf-8")
    assert main(["verify-quote", "--kind", "tdx", path, "--dcap-only"]) == 0
    out = capsys.readouterr().out
    assert "scope     : dcap-only" in out and "verified  : True" in out


def test_vq_tdx_raw_quote_with_dcap_only_is_scoped_the_same(capsys):
    rc = main(["verify-quote", "--kind", "tdx", str(FIXTURES / "tdx_quote_gcp.json"), "--dcap-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "scope     : dcap-only" in out and "verified  : True" in out


def test_vq_dcap_only_is_refused_for_other_kinds(capsys):
    rc = main(["verify-quote", "--kind", "gpu", str(FIXTURES / "gpu_h100_attestation.json"), "--dcap-only"])
    assert rc == 2
    assert "--dcap-only applies only to --kind tdx" in capsys.readouterr().err


def test_vq_tdx_azure_vtpm_bundle_cannot_supply_its_own_root_pin(capsys, tmp_path):
    import base64
    import hashlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_tdx import build_quote

    quote, root = build_quote()
    fingerprint = hashlib.sha256(root.public_bytes(serialization.Encoding.DER)).hexdigest()
    path = _azure_vtpm_bundle(
        tmp_path,
        kind="wcm-azure-tdx-vtpm/v1",
        dcap_b64=base64.b64encode(quote).decode(),
        outer={"intel_sgx_root_ca_sha256": fingerprint},
    )
    assert main(["verify-quote", "--kind", "tdx", path, "--dcap-only"]) == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out and "pinned Intel SGX root" in out


def test_vq_tdx_rejects_unknown_bundle_kind(capsys, tmp_path):
    path = _azure_vtpm_bundle(tmp_path, kind="wcm-azure-snp-vtpm/v1", dcap_b64=_azure_dcap_b64())
    assert main(["verify-quote", "--kind", "tdx", path]) == 1
    assert "unexpected Azure TDX vTPM bundle kind" in capsys.readouterr().out


def test_vq_tdx_malformed_azure_vtpm_bundle_is_a_denial(capsys, tmp_path):
    import base64

    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"quote_b64": base64.b64encode(b"{not json").decode()}))
    assert main(["verify-quote", "--kind", "tdx", str(path)]) == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out and "invalid Azure vTPM evidence" in out


def test_vq_tdx_azure_vtpm_bundle_with_garbage_dcap_quote_is_a_denial(capsys, tmp_path):
    import base64

    path = _azure_vtpm_bundle(
        tmp_path, kind="wcm-azure-tdx-vtpm/v1", dcap_b64=base64.b64encode(b"\0" * 3).decode()
    )
    assert main(["verify-quote", "--kind", "tdx", path]) == 1
    out = capsys.readouterr().out
    assert "verified  : False" in out and "TDX quote too short" in out


# -- verify-quote: GPU ---------------------------------------------------------


def test_vq_gpu_ok(capsys):
    assert main(["verify-quote", "--kind", "gpu", str(FIXTURES / "gpu_h100_attestation.json")]) == 0
    out = capsys.readouterr().out
    assert "verified  : True" in out and "GSP FMC LF" in out


def test_vq_gpu_wrong_nonce(capsys):
    rc = main(
        ["verify-quote", "--kind", "gpu", str(FIXTURES / "gpu_h100_attestation.json"), "--nonce", BAD_NONCE]
    )
    assert rc == 1
    assert "verified  : False" in capsys.readouterr().out


def test_vq_gpu_untrusted_root(capsys, tmp_path):
    rc = main(
        [
            "verify-quote",
            "--kind",
            "gpu",
            str(FIXTURES / "gpu_h100_attestation.json"),
            "--root",
            _stranger_root(tmp_path),
        ]
    )
    assert rc == 1
    assert "trusted root" in capsys.readouterr().out.lower()


# -- gate ----------------------------------------------------------------------


def test_gate_conformant_releases(capsys, example_dict, tmp_path):
    rc = main(["gate", _manifest_file(tmp_path, example_dict)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "released        : True" in out
    assert "[PASS] serving_image" in out
    # Mock evidence cannot be verified; the diagnostic says so rather than
    # reporting a refusal a KBS with verifiers would not make.
    for name in ("cpu_quote_verified", "gpu_report_verified", "gpu"):
        assert f"[SKIP] {name}" in out


def test_gate_skip_does_not_hide_a_gpu_failure(capsys, example_dict, tmp_path):
    rc = main(["gate", _manifest_file(tmp_path, example_dict), "--gpu-measurement", "wrong"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] gpu" in out


def test_gate_bad_serving_image_refuses(capsys, example_dict, tmp_path):
    rc = main(
        ["gate", _manifest_file(tmp_path, example_dict), "--serving-image", "sha256:" + "0" * 64]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "released        : False" in out
    assert "[FAIL] serving_image" in out


# -- inspect -------------------------------------------------------------------


def test_inspect_prints_key_fields(capsys, example_dict, tmp_path):
    assert main(["inspect", _manifest_file(tmp_path, example_dict)]) == 0
    out = capsys.readouterr().out
    assert example_dict["weights_hash"] in out
    assert "base_confidentiality" in out
    assert "release_policy" in out


# -- verify-provenance (model-signing extra) -----------------------------------


@requires_ms
def test_verify_provenance_cli_round_trip(capsys, example_dict, tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ec as _ec
    from wcm import model_signing_digest

    priv = _ec.generate_private_key(_ec.SECP256R1())
    priv_p = tmp_path / "key.pem"
    priv_p.write_bytes(
        priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    pub_p = tmp_path / "key.pub"
    pub_p.write_bytes(
        priv.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    model = tmp_path / "model.bin"
    model.write_bytes(b"weights-v1")
    sig = tmp_path / "model.sig"
    model_signing.signing.Config().use_elliptic_key_signer(private_key=str(priv_p)).sign(
        str(model), str(sig)
    )

    doc = copy.deepcopy(example_dict)
    doc["provenance"] = {"model_signing": {"signed_digest": model_signing_digest(str(model))}}
    manifest = _manifest_file(tmp_path, doc)

    rc = main(
        [
            "verify-provenance",
            manifest,
            "--model",
            str(model),
            "--signature",
            str(sig),
            "--public-key",
            str(pub_p),
        ]
    )
    assert rc == 0
    assert "verified  : True" in capsys.readouterr().out
