// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, RefObject, SetStateAction, useCallback, useEffect, useRef, useState } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Button, dimensionValue, Flex, toast } from '@geti/ui';
import { clsx } from 'clsx';

import { useWebRTCConnection } from '../../../components/stream/web-rtc-connection-provider';
import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import { useEventListener } from '../../../hooks/event-listener/event-listener.hook';
import { Fps } from './fps/fps.component';

import classes from './stream.module.scss';

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

export const Stream = () => {
    const videoRef = useStreamToVideo();
    const { projectId } = useProjectIdentifier();
    const [hasCaptureAnimation, setHasCaptureAnimation] = useState(false);
    const [size, setSize] = useState({ height: 608, width: 892 });

    useSetTargetSizeBasedOnVideo(setSize, videoRef);
    useEventListener('animationend', () => setHasCaptureAnimation(false), videoRef);

    const captureImageMutation = $api.useMutation('get', '/api/projects/{project_id}/capture', {
        onError: () => {
            toast({ type: 'error', message: `Failed to upload 1 item` });
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/images', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const handleCaptureFrame = async () => {
        setHasCaptureAnimation(true);

        await captureImageMutation.mutateAsync({
            params: { path: { project_id: projectId } },
        });
    };

    return (
        <Flex
            position={'relative'}
            direction={'column'}
            alignItems={'center'}
            justifyContent={'center'}
            UNSAFE_style={{ width: '100%', height: '100%', paddingBlockEnd: dimensionValue('size-400') }}
        >
            <Fps projectId={projectId} />

            <ZoomTransform target={size}>
                <video
                    ref={videoRef}
                    muted
                    autoPlay
                    playsInline
                    width={size.width}
                    height={size.height}
                    controls={false}
                    aria-label='stream player'
                    style={{ background: 'var(--spectrum-global-color-gray-200)' }}
                    className={clsx({ [classes.takeOldCamera]: hasCaptureAnimation })}
                />
            </ZoomTransform>
            <Button onPress={handleCaptureFrame} variant='primary' UNSAFE_className={classes.captureButton}>
                Capture
            </Button>
        </Flex>
    );
};
