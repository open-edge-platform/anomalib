// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { Size } from 'react-aria-components';

import { VirtualizerGridLayout } from './virtualizer-grid-layout.component';

// required configuration; otherwise, the list renders empty
const mockedLayoutOptions = {
    maxColumns: 1,
    maxItemSize: new Size(100, 100),
    maxHorizontalSpace: 1000,
};

describe('VirtualizerGridLayout', () => {
    const mockedItems = [
        { id: '1', name: 'Item 1' },
        { id: '2', name: 'Item 2' },
        { id: '3', name: 'Item 3' },
    ];

    it('renders all items as visible options', () => {
        const mockedLoadingMore = vi.fn();

        render(
            <VirtualizerGridLayout
                items={mockedItems}
                ariaLabel={'test list'}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={false}
                onLoadMore={mockedLoadingMore}
                contentItem={(item) => <div>{item.name}</div>}
            />
        );

        expect(mockedLoadingMore).toHaveBeenCalledOnce();
        mockedItems.forEach((item) => expect(screen.getByRole('option', { name: item.name })).toBeVisible());
    });

    it('renders with empty items', () => {
        render(
            <VirtualizerGridLayout
                items={[]}
                ariaLabel={'empty list'}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={false}
                onLoadMore={vi.fn()}
                contentItem={() => <div></div>}
            />
        );
        expect(screen.queryAllByRole('option')).toHaveLength(0);
    });

    it('renders loading indicator when isLoadingMore is true', () => {
        const mockedLoadingMore = vi.fn();
        render(
            <VirtualizerGridLayout
                items={mockedItems}
                ariaLabel={'loading list'}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={true}
                onLoadMore={mockedLoadingMore}
                contentItem={(item) => <div>{item.name}</div>}
            />
        );

        expect(mockedLoadingMore).not.toHaveBeenCalled();
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
});
