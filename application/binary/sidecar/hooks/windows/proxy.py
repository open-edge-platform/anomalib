# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime hook: Windows proxy auto-detection via WinHTTP."""

import ctypes
import os
from ctypes import wintypes

# Constants
WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY = 4
WINHTTP_NO_PROXY_NAME = None
WINHTTP_NO_PROXY_BYPASS = None
WINHTTP_FLAG_SYNC = 0x00000000

WINHTTP_AUTOPROXY_AUTO_DETECT = 0x00000001
WINHTTP_AUTO_DETECT_TYPE_DHCP = 0x00000001
WINHTTP_AUTO_DETECT_TYPE_DNS_A = 0x00000002


class WINHTTP_AUTOPROXY_OPTIONS(ctypes.Structure):
    _fields_ = [
        ("dwFlags", wintypes.DWORD),
        ("dwAutoDetectFlags", wintypes.DWORD),
        ("lpszAutoConfigUrl", wintypes.LPWSTR),
        ("lpvReserved", wintypes.LPVOID),
        ("dwReserved", wintypes.DWORD),
        ("fAutoLogonIfChallenged", wintypes.BOOL),
    ]


class WINHTTP_PROXY_INFO(ctypes.Structure):
    _fields_ = [
        ("dwAccessType", wintypes.DWORD),
        ("lpszProxy", wintypes.LPWSTR),
        ("lpszProxyBypass", wintypes.LPWSTR),
    ]


WinHttpOpen = ctypes.windll.winhttp.WinHttpOpen  # type: ignore[attr-defined]
WinHttpOpen.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
WinHttpOpen.restype = wintypes.HANDLE

WinHttpGetProxyForUrl = ctypes.windll.winhttp.WinHttpGetProxyForUrl  # type: ignore[attr-defined]
WinHttpGetProxyForUrl.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCWSTR,
    ctypes.POINTER(WINHTTP_AUTOPROXY_OPTIONS),
    ctypes.POINTER(WINHTTP_PROXY_INFO),
]
WinHttpGetProxyForUrl.restype = wintypes.BOOL

WinHttpCloseHandle = ctypes.windll.winhttp.WinHttpCloseHandle  # type: ignore[attr-defined]
WinHttpCloseHandle.argtypes = [wintypes.HANDLE]
WinHttpCloseHandle.restype = wintypes.BOOL


def _normalize_proxy_url(proxy: str) -> str:
    """Normalize WinHTTP proxy value to a URL with scheme for HTTP_PROXY/HTTPS_PROXY.

    WinHTTP may return "host:port", "host", or "http=host:port;https=host:port".
    Python HTTP clients expect a full URL with scheme (e.g. http://host:port).

    Args:
        proxy: Raw proxy string from WinHTTP (possibly semicolon-separated, no scheme).

    Returns:
        A single proxy URL with http:// scheme, suitable for HTTP_PROXY/HTTPS_PROXY.
    """
    proxy = proxy.strip()
    if not proxy:
        return proxy
    # Use first proxy if semicolon-separated (e.g. "proxy1:8080;proxy2:8080" or "http=host:80;https=host:80")
    first = proxy.split(";")[0].strip()
    # If WinHTTP used protocol-prefix form (e.g. "http=host:80"), use the part after "="
    if "=" in first:
        first = first.split("=", 1)[1].strip()
    # If it already has a scheme, use as-is (after stripping)
    if first.lower().startswith(("http://", "https://")):
        return first
    # Otherwise prepend http:// so clients get a valid URL
    return "http://" + first


def _proxy_resolver(url: str) -> str | None:
    # Open WinHTTP session
    hSession = WinHttpOpen(
        "PythonProxyResolver",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        WINHTTP_FLAG_SYNC,
    )
    if not hSession:
        return None

    try:
        options = WINHTTP_AUTOPROXY_OPTIONS()
        options.dwFlags = WINHTTP_AUTOPROXY_AUTO_DETECT
        options.dwAutoDetectFlags = WINHTTP_AUTO_DETECT_TYPE_DHCP | WINHTTP_AUTO_DETECT_TYPE_DNS_A
        options.lpszAutoConfigUrl = None
        options.fAutoLogonIfChallenged = True

        proxy_info = WINHTTP_PROXY_INFO()
        success = WinHttpGetProxyForUrl(hSession, url, ctypes.byref(options), ctypes.byref(proxy_info))
        if not success or not proxy_info.lpszProxy:
            return None
        return _normalize_proxy_url(proxy_info.lpszProxy)
    finally:
        WinHttpCloseHandle(hSession)


print("Setup Hook: Detecting proxy")
proxy = _proxy_resolver("https://huggingface.co")
print("Setup Hook: Detected proxy: ", proxy)

if proxy:
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
