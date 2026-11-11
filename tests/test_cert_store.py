"""
The app must verify HTTPS against the Windows certificate store. On networks
that inspect HTTPS, the certifi list rejects every connection, so Claude and
Hugging Face calls fail even with working internet. No network calls here.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_app_switches_https_to_the_system_certificate_store():
    # Run in a fresh interpreter so this test can't leak the global ssl patch
    # into other tests, and so it checks what actually happens on app import.
    code = (
        "import ssl, sys; sys.path.insert(0, '.');\n"
        "import app\n"
        "import truststore\n"
        "print(ssl.SSLContext is truststore.SSLContext)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.stdout.strip().splitlines()[-1] == "True", result.stderr[-2000:]


def test_truststore_is_pinned_in_requirements():
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("truststore==") for line in reqs)
