// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook, waitFor } from '@testing-library/react';
import { getMockedMediaItem } from 'mocks/mock-media-item';
import { useQueryState } from 'nuqs';

import { useGetMediaItems } from './use-get-media-items.hook';
import { useSelectedMediaItem } from './use-selected-media-item.hook';

vi.mock('./use-get-media-items.hook', () => ({
    useGetMediaItems: vi.fn(),
}));

vi.mock('nuqs', async () => {
    const actual = await vi.importActual('nuqs');
    return {
        ...actual,
        useQueryState: vi.fn(),
    };
});

describe('useSelectedMediaItem', () => {
    const mockMediaItems = [
        getMockedMediaItem({ id: 'media-1', filename: 'image-1.jpg' }),
        getMockedMediaItem({ id: 'media-2', filename: 'image-2.jpg' }),
        getMockedMediaItem({ id: 'media-3', filename: 'image-3.jpg' }),
    ];

    const renderApp = (props: Partial<ReturnType<typeof useGetMediaItems>> & { selectedMediaItem: string | null }) => {
        vi.mocked(useGetMediaItems).mockReturnValue({
            isLoading: false,
            hasNextPage: false,
            mediaItems: mockMediaItems,
            isFetchingNextPage: false,
            fetchNextPage: vi.fn(),
            ...props,
        });

        vi.mocked(useQueryState).mockReturnValue([props?.selectedMediaItem ?? null, vi.fn()]);

        renderHook(() => useSelectedMediaItem());
    };

    it('does not fetch next page when selectedMediaItemId is null', () => {
        const mockFetchNextPage = vi.fn();

        renderApp({ hasNextPage: true, fetchNextPage: mockFetchNextPage, selectedMediaItem: null });

        expect(mockFetchNextPage).not.toHaveBeenCalled();
    });

    it('does not fetch next page when selectedMediaItem is found', () => {
        const mockFetchNextPage = vi.fn();
        const selectedMediaItem = String(mockMediaItems[0].id);
        renderApp({
            hasNextPage: true,
            fetchNextPage: mockFetchNextPage,
            mediaItems: mockMediaItems,
            selectedMediaItem,
        });

        expect(mockFetchNextPage).not.toHaveBeenCalled();
    });

    it('fetches next page when selectedMediaItem is not found and hasNextPage is true', async () => {
        const mockFetchNextPage = vi.fn();

        renderApp({ hasNextPage: true, fetchNextPage: mockFetchNextPage, selectedMediaItem: 'media-999' });

        await waitFor(() => {
            expect(mockFetchNextPage).toHaveBeenCalled();
        });
    });

    it('does not fetch next page when isFetchingNextPage is true', () => {
        const mockFetchNextPage = vi.fn();

        renderApp({
            hasNextPage: true,
            isFetchingNextPage: true,
            selectedMediaItem: 'media-999',
            fetchNextPage: mockFetchNextPage,
        });

        expect(mockFetchNextPage).not.toHaveBeenCalled();
    });

    it('does not fetch next page when hasNextPage is false', () => {
        const mockFetchNextPage = vi.fn();

        renderApp({ hasNextPage: false, fetchNextPage: mockFetchNextPage, selectedMediaItem: 'media-999' });

        expect(mockFetchNextPage).not.toHaveBeenCalled();
    });
});
