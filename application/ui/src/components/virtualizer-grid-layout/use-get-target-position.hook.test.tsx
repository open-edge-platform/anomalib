// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from '@testing-library/react';

import { useGetTargetPosition } from './use-get-target-position.hook';

const getMockedRef = (element: HTMLDivElement | null) => {
    return { current: element } as unknown as React.RefObject<HTMLDivElement>;
};
describe('useGetTargetPosition', () => {
    const mockCallback = vi.fn();

    beforeEach(() => {
        vi.useFakeTimers();
        vi.clearAllMocks();
        vi.clearAllTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('should not call callback when container is not provided', () => {
        renderHook(() =>
            useGetTargetPosition({
                gap: 10,
                ref: getMockedRef(null),
                scrollToIndex: 5,
                callback: mockCallback,
            })
        );

        vi.advanceTimersByTime(500);

        expect(mockCallback).not.toHaveBeenCalled();
    });

    it('calls callback with correct scroll position', () => {
        const gap = 0;
        const childWidth = 100;
        const childHeight = 100;
        const scrollToIndex = 12;
        const containerWidth = 200;

        const container = document.createElement('div');
        const firstChild = document.createElement('div');
        Object.defineProperty(firstChild, 'clientWidth', { value: containerWidth });

        const child = document.createElement('div');
        Object.defineProperty(child, 'clientWidth', { value: childWidth });
        Object.defineProperty(child, 'clientHeight', { value: childHeight });

        container.appendChild(firstChild);
        firstChild.appendChild(child);

        const itemsPerRow = Math.floor(containerWidth / childWidth); // 2
        const targetRow = Math.floor(scrollToIndex / itemsPerRow); // 6
        const expectedScrollPos = (childHeight + gap) * targetRow; // 600

        renderHook(() =>
            useGetTargetPosition({
                gap,
                ref: getMockedRef(container),
                scrollToIndex,
                callback: mockCallback,
            })
        );

        vi.advanceTimersByTime(500);

        expect(mockCallback).toHaveBeenCalledWith(expectedScrollPos);
    });

    it('returns zero when container has no children', () => {
        const container = document.createElement('div');
        const firstChild = document.createElement('div');
        Object.defineProperty(firstChild, 'clientWidth', { value: 1000 });
        container.appendChild(firstChild);

        renderHook(() =>
            useGetTargetPosition({
                gap: 10,
                ref: getMockedRef(container),
                scrollToIndex: 5,
                callback: mockCallback,
            })
        );

        vi.advanceTimersByTime(500);

        expect(mockCallback).toHaveBeenCalledWith(0);
    });

    describe('should not call callback with invalid index', () => {
        it.each([undefined, null, -1, 1.5, NaN])('scrollToIndex: %p', (invalidIndex) => {
            renderHook(() =>
                useGetTargetPosition({
                    gap: 10,
                    ref: getMockedRef(document.createElement('div')),
                    scrollToIndex: invalidIndex as number,
                    callback: mockCallback,
                })
            );

            vi.advanceTimersByTime(500);

            expect(mockCallback).not.toHaveBeenCalled();
        });
    });
});
