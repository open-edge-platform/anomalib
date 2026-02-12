// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, RefObject, SetStateAction, useCallback, useEffect, useRef, useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { Button, dimensionValue, Flex, toast } from '@geti/ui';
import { clsx } from 'clsx';

import { useStreamConnection } from '../../../components/stream/stream-connection-provider';
import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import { useEventListener } from '../../../hooks/event-listener/event-listener.hook';
import { Fps } from './fps/fps.component';

import classes from './stream.module.scss';

const useSetTargetSizeBasedOnImage = (
    setSize: Dispatch<SetStateAction<{ width: number; height: number }>>,
    imageRef: RefObject<HTMLImageElement | null>,
    streamUrl: string | null
) => {
    useEffect(() => {
        const image = imageRef.current;
        if (!image) return;

        const updateSize = () => {
            if (image.naturalWidth && image.naturalHeight) {
                setSize({ width: image.naturalWidth, height: image.naturalHeight });
            }
        };

        const resizeObserver = new ResizeObserver(updateSize);
        image.addEventListener('load', updateSize);
        resizeObserver.observe(image);

        return () => {
            image.removeEventListener('load', updateSize);
            resizeObserver.disconnect();
        };
    }, [setSize, imageRef, streamUrl]);

    useEffect(() => {
        const image = imageRef.current;
        if (!image || !streamUrl) {
            return;
        }

        let attempts = 0;
        const maxAttempts = 20;
        const interval = setInterval(() => {
            if (image.naturalWidth && image.naturalHeight) {
                setSize({ width: image.naturalWidth, height: image.naturalHeight });
                clearInterval(interval);
                return;
            }

            attempts += 1;
            if (attempts >= maxAttempts) {
                clearInterval(interval);
            }
        }, 200);

        return () => {
            clearInterval(interval);
        };
    }, [setSize, imageRef, streamUrl]);
};

export const Stream = () => {
    const imageRef = useRef<HTMLImageElement>(null);
    const { projectId } = useProjectIdentifier();
    const { setStatus, status, streamUrl } = useStreamConnection();
    const [hasCaptureAnimation, setHasCaptureAnimation] = useState(false);
    const [size, setSize] = useState({ height: 608, width: 892 });

    useSetTargetSizeBasedOnImage(setSize, imageRef, streamUrl);
    useEventListener('animationend', () => setHasCaptureAnimation(false), imageRef);

    const handleStreamLoad = useCallback(() => {
        setStatus('connected');
    }, [setStatus]);

    const handleStreamError = useCallback(() => {
        // Only set failed if we were previously connected or connecting,
        // and avoid flipping to failed if the url was just cleared (null)
        if (streamUrl) {
            setStatus((current) => (current === 'connected' ? 'disconnected' : 'failed'));
            toast({ type: 'error', message: 'Stream connection failed' });
        }
    }, [setStatus, streamUrl]);

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
            {status === 'connected' && <Fps projectId={projectId} />}

            <ZoomTransform target={size}>
                <img
                    key={streamUrl}
                    ref={imageRef}
                    src={streamUrl ?? undefined}
                    width={size.width}
                    height={size.height}
                    aria-label='stream player'
                    alt='stream'
                    onLoad={handleStreamLoad}
                    onError={handleStreamError}
                    style={{ background: 'var(--spectrum-global-color-gray-200)' }}
                    className={clsx({ [classes.takeOldCamera]: hasCaptureAnimation })}
                />
            </ZoomTransform>
            {status === 'connected' && (
                <Button onPress={handleCaptureFrame} variant='primary' UNSAFE_className={classes.captureButton}>
                    Capture
                </Button>
            )}
        </Flex>
    );
};
