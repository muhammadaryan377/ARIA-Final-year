"""Encrypt/decrypt secrets at rest with Windows DPAPI.

DPAPI (CryptProtectData) binds the ciphertext to the current Windows user
account, so a stolen `.env` file is unreadable on any other machine or user.
Keys are stored as `enc1:<base64>` values; `decrypt_env_value()` is called
from core.config right after load_dotenv so `os.getenv` sees the real key.

Platform honesty: only Windows DPAPI encrypts. On Linux/macOS there is no
local trust anchor, so `encrypt()` returns plaintext AND warns loudly;
`is_encrypted()` reports False so callers never claim "encrypted at rest"
when the value is plaintext. Unix secret files must additionally carry
0600 permissions (see `ensure_private_file`); plaintext values are never
written to logs (see `redact`).
"""

import base64
import ctypes
import os
import sys
import warnings

PREFIX = "enc1:"


def is_encrypted(value) -> bool:
    """Return True only when the value carries the DPAPI `enc1:` envelope."""
    return isinstance(value, str) and value.startswith(PREFIX)


def redact(value, keep: int = 0) -> str:
    """Return a log-safe placeholder for a secret (never the plaintext)."""
    if not value:
        return "[missing]"
    if keep and isinstance(value, str) and len(value) > keep + 3:
        return value[:keep] + "…[redacted]"
    return "[redacted]"


def ensure_private_file(path) -> None:
    """Best-effort private perms on a secret-bearing file.

    POSIX: chmod 0600 (unchanged). Windows: restrict the file ACL to the
    current user only via icacls (fail-open with warning, never crashes
    startup, never logs secret values).
    """
    try:
        if sys.platform != "win32":
            os.chmod(path, 0o600)
            return
        # Windows branch: current-user-only ACL via icacls (no shell).
        try:
            import ctypes  # noqa: F401 - confirms Win32 API surface present
        except Exception:
            pass
        try:
            import logging as _logging
            import subprocess as _subprocess

            _username = (os.getenv("USERNAME") or "").strip()
            if not _username:
                try:
                    import getpass as _getpass

                    _username = (_getpass.getuser() or "").strip()
                except Exception:
                    _username = ""
            if not _username:
                _logging.getLogger("uvicorn.error").warning(
                    "core.secret_box: cannot determine current user; "
                    "leaving file ACL unchanged."
                )
                return
            try:
                _subprocess.run(
                    [
                        "icacls",
                        str(path),
                        "/inheritance:r",
                        "/grant:r",
                        f"{_username}:F",
                    ],
                    shell=False,
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except (FileNotFoundError, OSError) as exc:
                _logging.getLogger("uvicorn.error").warning(
                    "core.secret_box: icacls unavailable (%s); "
                    "leaving file ACL unchanged.",
                    type(exc).__name__,
                )
            except Exception as exc:
                _logging.getLogger("uvicorn.error").warning(
                    "core.secret_box: Windows ACL hardening failed (%s); "
                    "leaving file ACL unchanged.",
                    type(exc).__name__,
                )
        except Exception:
            pass
    except Exception:
        pass

if sys.platform == "win32":
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def _blob(data: bytes) -> DATA_BLOB:
        buf = ctypes.create_string_buffer(data, len(data))
        blob = DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        # Keep the buffer alive: without this reference the bytes are freed
        # when _blob returns, leaving a dangling pointer (use-after-free that
        # fails intermittently, typically in long-lived server processes).
        blob._keepalive = buf
        return blob

    def _protect(data: bytes) -> bytes:
        blob_in = _blob(data)
        blob_out = DATA_BLOB()
        if not ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(blob_in), "ARIA secret", None, None, None, 0, ctypes.byref(blob_out)
        ):
            raise OSError("CryptProtectData failed")
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)

    def _unprotect(data: bytes) -> bytes:
        blob_in = _blob(data)
        blob_out = DATA_BLOB()
        if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
        ):
            raise OSError("CryptUnprotectData failed")
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)

    def encrypt(plaintext: str) -> str:
        return PREFIX + base64.b64encode(_protect(plaintext.encode("utf-8"))).decode("ascii")

    def decrypt(ciphertext: str) -> str:
        if not ciphertext.startswith(PREFIX):
            return ciphertext
        raw = base64.b64decode(ciphertext[len(PREFIX):])
        return _unprotect(raw).decode("utf-8")

else:
    # Non-Windows fallback: NO encryption is available (no local trust
    # anchor). Return plaintext but warn loudly every time so operators never
    # mistake it for encrypted storage. Use file permissions (0600 via
    # ensure_private_file) as the safest existing store on this platform.
    def encrypt(plaintext: str) -> str:
        warnings.warn(
            "core.secret_box: no OS keystore on this platform; secret stored "
            "as PLAINTEXT (not encrypted). Restrict file permissions (0600) "
            "and prefer Windows DPAPI or a managed secret store in production.",
            RuntimeWarning,
            stacklevel=2,
        )
        return plaintext

    def decrypt(ciphertext: str) -> str:
        # Never raise on plaintext passthrough; never log the value.
        return ciphertext