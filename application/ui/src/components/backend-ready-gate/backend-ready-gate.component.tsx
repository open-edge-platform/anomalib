// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Button, Heading, IllustratedMessage, IntelBrandedLoading, Text, View } from '@geti-ui/ui';
import { NotFound } from '@geti-ui/ui/icons';
import { useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';

// The desktop (Tauri) app starts the backend sidecar together with the UI, and the
// backend can take a while to boot (especially on Windows). Keep polling for
// roughly two minutes before giving up and showing an error with a retry button.
export const BACKEND_READY_RETRY_DELAY_MS = 1000;
export const BACKEND_READY_MAX_RETRIES = 120;

export const BACKEND_READY_QUERY_KEY = ['backend-ready'] as const;

const checkBackendReady = async (): Promise<true> => {
    // openapi-fetch only throws when the request itself fails (e.g. the server is not
    // listening yet). Any HTTP response, even an error status, means the backend is up.
    await fetchClient.GET('/api/projects');

    return true;
};

interface BackendReadyGateProps {
    children: ReactNode;
    retryDelayMs?: number;
    maxRetries?: number;
}

export const BackendReadyGate = ({
    children,
    retryDelayMs = BACKEND_READY_RETRY_DELAY_MS,
    maxRetries = BACKEND_READY_MAX_RETRIES,
}: BackendReadyGateProps) => {
    const { isPending, isError, isFetching, refetch } = useQuery({
        queryKey: BACKEND_READY_QUERY_KEY,
        queryFn: checkBackendReady,
        retry: maxRetries,
        retryDelay: retryDelayMs,
        staleTime: Infinity,
        gcTime: Infinity,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
    });

    if (isError && !isFetching) {
        return (
            <View height={'100vh'}>
                <IllustratedMessage>
                    <NotFound />
                    <Heading>Could not connect to the Anomalib Studio server</Heading>
                    <Text>The server did not start in time. Make sure it is running and try again.</Text>
                    <Button variant={'accent'} marginTop={'size-200'} onPress={() => refetch()}>
                        Try again
                    </Button>
                </IllustratedMessage>
            </View>
        );
    }

    if (isPending || isError) {
        return <IntelBrandedLoading />;
    }

    return <>{children}</>;
};
