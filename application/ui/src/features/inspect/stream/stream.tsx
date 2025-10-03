// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, RefObject, SetStateAction, useCallback, useEffect, useRef } from 'react';

import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { ZoomTransform } from '../../../components/zoom/zoom-transform';

const useSetTargetSizeBasedOnVideo = (
    setSize: Dispatch<SetStateAction<{ width: number; height: number }>>,
    videoRef: RefObject<HTMLVideoElement | null>
) => {
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const onLoaded = () => {
            if (video.videoWidth && video.videoHeight) {
                setSize({ width: video.videoWidth, height: video.videoHeight });
            }
        };

        const resizeObserver = new ResizeObserver(() => {
            if (video.videoWidth && video.videoHeight) {
                setSize({ width: video.videoWidth, height: video.videoHeight });
            }
        });

        video.addEventListener('loadedmetadata', onLoaded);
        resizeObserver.observe(video);

        return () => {
            video.removeEventListener('loadedmetadata', onLoaded);
            resizeObserver.disconnect();
        };
    }, [setSize, videoRef]);
};

const useStreamToVideo = () => {
    const videoRef = useRef<HTMLVideoElement>(null);

    const { status, webRTCConnectionRef } = useWebRTCConnection();

    const connect = useCallback(async () => {
        const videoOutput = videoRef.current;
        const webrtcConnection = webRTCConnectionRef.current;
        const peerConnection = webrtcConnection?.getPeerConnection();

        if (!peerConnection) {
            return;
        }

        const receivers = peerConnection.getReceivers() ?? [];
        const stream = new MediaStream(receivers.map((receiver) => receiver.track).filter(Boolean));

        if (videoOutput && videoOutput.srcObject !== stream) {
            videoOutput.srcObject = stream;
        }
    }, [videoRef, webRTCConnectionRef]);

    useEffect(() => {
        if (status === 'connected') {
            connect();
        }
    }, [status, connect]);

    useEffect(() => {
        const webrtcConnection = webRTCConnectionRef.current;
        const peerConnection = webrtcConnection?.getPeerConnection();

        if (!peerConnection) {
            return;
        }

        peerConnection.addEventListener('track', connect);

        return () => {
            peerConnection.removeEventListener('track', connect);
        };
    }, [webRTCConnectionRef, connect]);

    return videoRef;
};

export const Stream = ({
    size,
    setSize,
}: {
    size: { width: number; height: number };
    setSize: Dispatch<SetStateAction<{ width: number; height: number }>>;
}) => {
    const videoRef = useStreamToVideo();

    useSetTargetSizeBasedOnVideo(setSize, videoRef);

    const { status } = useWebRTCConnection();

    return (
        <ZoomTransform target={size}>
            <div style={{ gridArea: 'innercanvas' }}>
                {status === 'connected' && (
                    // eslint-disable-next-line jsx-a11y/media-has-caption
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        width={size.width}
                        height={size.height}
                        controls={false}
                        style={{
                            background: 'var(--spectrum-global-color-gray-200)',
                        }}
                    />
                )}
            </div>
        </ZoomTransform>
    );
};
