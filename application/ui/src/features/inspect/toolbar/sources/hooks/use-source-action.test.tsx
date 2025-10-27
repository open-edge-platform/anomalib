// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { startTransition } from 'react';

import { act, renderHook, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { TestProviders } from 'src/providers';

import { ImagesFolderSourceConfig } from '../util';
import { useSourceAction } from './use-source-action.hook';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

const mockedConfig: ImagesFolderSourceConfig = {
    id: 'images_folder-id',
    name: 'Test Folder',
    source_type: 'images_folder',
    images_folder_path: '/path/to/images',
    ignore_existing_images: false,
};

const bodyFormatter = (formData: FormData) => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'images_folder' as const,
    images_folder_path: String(formData.get('images_folder_path')),
    ignore_existing_images: formData.get('ignore_existing_images') === 'on' ? true : false,
});

const renderApp = async ({
    isNewSource = false,
    config = mockedConfig,
    newResource = () => HttpResponse.error(),
    updateResource = () => HttpResponse.error(),
}) => {
    server.use(
        http.post('/api/sources', newResource),
        http.patch('/api/sources/{source_id}', updateResource),
        http.patch('/api/projects/{project_id}/pipeline', () => HttpResponse.json({}))
    );

    const { result } = renderHook(() => useSourceAction({ config: mockedConfig, isNewSource, bodyFormatter }), {
        wrapper: TestProviders,
    });
    const [_state, submitAction] = result.current;

    const formData = new FormData();
    formData.append('name', config.name);
    formData.append('source_type', config.source_type);
    formData.append('images_folder_path', config.images_folder_path);
    formData.append('ignore_existing_images', String(config.ignore_existing_images));
    config.id && formData.append('id', config.id);

    await act(async () => {
        startTransition(async () => submitAction(formData));
    });

    return result.current;
};

describe('useSourceAction', () => {
    it('return initial config', () => {
        const { result } = renderHook(
            () => useSourceAction({ config: mockedConfig, isNewSource: true, bodyFormatter }),
            { wrapper: TestProviders }
        );

        expect(result.current[0]).toEqual(mockedConfig);
    });

    describe('new configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';
            await renderApp({
                isNewSource: true,
                newResource: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save source configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submit new image folder config and show success message', async () => {
            const mockedNewItemId = 'new-id-test';
            const [state] = await renderApp({
                isNewSource: true,
                newResource: () => HttpResponse.json({ ...mockedConfig, id: mockedNewItemId }),
            });

            await waitFor(() => {
                expect(screen.getByText('Source configuration created successfully.')).toBeVisible();
                expect(state.id).toBe(mockedNewItemId);
            });
        });
    });

    describe('edit configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';

            await renderApp({
                isNewSource: false,
                updateResource: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save source configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submits folder config and show success message', async () => {
            const newConfig = { ...mockedConfig, id: 'mockedResponseId', name: 'Updated Name' };

            const [state] = await renderApp({
                isNewSource: false,
                config: newConfig,
                updateResource: () => HttpResponse.json(newConfig),
            });

            await waitFor(() => {
                expect(screen.getByText('Source configuration updated successfully.')).toBeVisible();
                expect(state.id).toBe(newConfig.id);
            });
        });
    });
});
