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
WinHttpGetProxyForUrl.restype = wintypes.DWORD

WinHttpCloseHandle = ctypes.windll.winhttp.WinHttpCloseHandle  # type: ignore[attr-defined]
WinHttpCloseHandle.argtypes = [wintypes.HANDLE]
WinHttpCloseHandle.restype = wintypes.BOOL


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
        result = WinHttpGetProxyForUrl(hSession, url, ctypes.byref(options), ctypes.byref(proxy_info))
        return proxy_info.lpszProxy if result == 1 else None
    finally:
        WinHttpCloseHandle(hSession)


print("Setup Hook: Detecting proxy")
proxy = _proxy_resolver("https://huggingface.co")
print("Setup Hook: Detected proxy: ", proxy)

if proxy:
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
