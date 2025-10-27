// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { TestProviders } from 'src/providers';

import { WebcamSourceConfig } from '../util';
import { useSourceMutation } from './use-source-mutation.hook';

const mockedSource: WebcamSourceConfig = {
    id: 'original-id',
    name: 'Mock Source',
    source_type: 'webcam' as const,
    device_id: 0,
};

describe('useSourceMutation', () => {
    it('creates a new source and return its resource id', async () => {
        const newResourceId = 'resource-id-123';
        const { result } = renderHook(() => useSourceMutation(true), {
            wrapper: TestProviders,
        });

        server.use(
            http.post('/api/sources', () => HttpResponse.json({ ...mockedSource, id: newResourceId })),
            http.patch('/api/sources/{source_id}', () => HttpResponse.error())
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(newResourceId);
        });
    });

    it('update a source item and returns its resource id', async () => {
        const { result } = renderHook(() => useSourceMutation(false), {
            wrapper: TestProviders,
        });

        server.use(
            http.post('/api/sources', () => HttpResponse.error()),
            http.patch('/api/sources/{source_id}', () => HttpResponse.json(mockedSource))
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(mockedSource.id);
        });
    });
});
