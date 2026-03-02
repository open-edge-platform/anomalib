# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import socket

from aiortc.sdp import SessionDescription

logger = logging.getLogger(__name__)


class SDPHandler:
    """Handler for SDP manipulation and processing."""

    async def resolve_hostname(self, hostname: str) -> str:
        """Resolve hostname to IP address.

        Args:
            hostname: The hostname to resolve.

        Returns:
            The resolved IP address or the original hostname if resolution fails.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, socket.gethostbyname, hostname)
        except Exception as exc:
            logger.warning(f"Failed to resolve hostname {hostname}: {exc}")
            return hostname

    async def mangle_sdp(self, sdp: str, advertise_ip: str | None) -> str:
        """Mangle SDP with advertise IP.

        Resolves hostname if necessary.

        Args:
            sdp: The Session Description Protocol string.
            advertise_ip: The IP address or hostname to advertise.

        Returns:
            The modified SDP string with the advertised IP.
        """
        if not advertise_ip:
            return sdp

        resolved_ip = await self.resolve_hostname(advertise_ip)
        return self._mangle_sdp_with_ip(sdp, resolved_ip)

    def _mangle_sdp_with_ip(self, sdp: str, ip: str) -> str:
        """Replace local IP addresses in SDP candidates and connection lines with the advertise IP.

        Useful for 1:1 NAT scenarios where STUN is not available.

        Args:
            sdp: The Session Description Protocol string.
            ip: The IP address to replace local IPs with.

        Returns:
            The modified SDP string.
        """
        try:
            parsed_sdp = SessionDescription.parse(sdp)
        except Exception as exc:
            logger.warning(f"Failed to parse SDP for mangling: {exc}. Returning original SDP.")
            return sdp

        if parsed_sdp.host:
            parsed_sdp.host = ip

        for media in parsed_sdp.media:
            if media.host:
                media.host = ip

            if media.rtcp_host:
                media.rtcp_host = ip

            for candidate in media.ice_candidates:
                if candidate.type == "host":
                    candidate.ip = ip

        return str(parsed_sdp)
