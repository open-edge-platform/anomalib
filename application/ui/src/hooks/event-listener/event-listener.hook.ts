// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { RefObject, useEffect, useLayoutEffect, useRef } from 'react';

function determineTargetElement<ElementType extends Element = Element>(
    element?: RefObject<ElementType | null> | ElementType | null
): ElementType | (Window & typeof globalThis) | null {
    if (element === undefined) {
        return window;
    }

    if (element === null) {
        return null;
    }

    if ('current' in element) {
        return element.current;
    }

    return element;
}

type EventType = GlobalEventHandlersEventMap & WindowEventHandlersEventMap & DocumentEventMap;

export function useEventListener<
    EventName extends keyof EventType,
    Handler extends (event: EventType[EventName]) => void,
    ElementType extends Element = Element,
>(eventName: EventName, handler: Handler, element?: RefObject<ElementType | null> | ElementType | null): void {
    const savedHandler = useRef<Handler>(handler);

    useLayoutEffect(() => {
        savedHandler.current = handler;
    }, [handler]);

    useEffect(() => {
        const controller = new AbortController();
        const targetElement = determineTargetElement(element);

        if (targetElement === null) {
            return;
        }

        targetElement.addEventListener(
            eventName,
            (event) => {
                if (savedHandler.current !== undefined) {
                    savedHandler.current(event as EventType[EventName]);
                }
            },
            {
                signal: controller.signal,
            }
        );

        return () => {
            controller.abort();
        };
    }, [eventName, element]);
}
