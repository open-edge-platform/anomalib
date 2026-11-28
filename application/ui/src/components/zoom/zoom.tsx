import { createContext, Dispatch, ReactNode, SetStateAction, useCallback, useContext, useRef, useState } from 'react';

import { isFunction } from 'lodash-es';

type ZoomState = { scale: number; translate: { x: number; y: number } };

const MIN_SCALE = 0.1;
const MAX_SCALE = 10.0;
const MIN_TRANSLATE = -5000;
const MAX_TRANSLATE = 5000;

const clamp = (val: number, min: number, max: number): number => Math.min(Math.max(val, min), max);

// Limits are widened to include the fit-to-container state so large images stay reachable.
const clampZoomState = (state: ZoomState, fit: ZoomState | null): ZoomState => {
    const minScale = Math.min(MIN_SCALE, fit?.scale ?? MIN_SCALE);
    const minX = Math.min(MIN_TRANSLATE, fit?.translate.x ?? 0);
    const maxX = Math.max(MAX_TRANSLATE, fit?.translate.x ?? 0);
    const minY = Math.min(MIN_TRANSLATE, fit?.translate.y ?? 0);
    const maxY = Math.max(MAX_TRANSLATE, fit?.translate.y ?? 0);

    return {
        scale: clamp(state.scale, minScale, MAX_SCALE),
        translate: {
            x: clamp(state.translate.x, minX, maxX),
            y: clamp(state.translate.y, minY, maxY),
        },
    };
};
export const Zoom = createContext<ZoomState>({
    scale: 1.0,
    translate: { x: 0, y: 0 },
});
const SetZoom = createContext<Dispatch<SetStateAction<ZoomState>> | null>(null);
const SetFitZoom = createContext<((state: ZoomState) => void) | null>(null);

export const useZoom = () => {
    return useContext(Zoom);
};

export const useSetZoom = () => {
    const context = useContext(SetZoom);

    if (!context) {
        throw new Error('');
    }

    return context;
};

export const useSetFitZoom = () => {
    const context = useContext(SetFitZoom);

    if (!context) {
        throw new Error('');
    }

    return context;
};

export const ZoomProvider = ({ children }: { children: ReactNode }) => {
    const [zoom, setZoom] = useState<ZoomState>({
        scale: 1.0,
        translate: { x: 0, y: 0 },
    });

    const fitZoom = useRef<ZoomState | null>(null);

    const setClampedZoom = useCallback<Dispatch<SetStateAction<ZoomState>>>((action) => {
        setZoom((prev) => {
            const next = isFunction(action) ? action(prev) : action;
            return clampZoomState(next, fitZoom.current);
        });
    }, []);

    const setFitZoom = useCallback((state: ZoomState) => {
        fitZoom.current = state;
        setZoom(state);
    }, []);

    return (
        <Zoom.Provider value={zoom}>
            <SetZoom.Provider value={setClampedZoom}>
                <SetFitZoom.Provider value={setFitZoom}>{children}</SetFitZoom.Provider>
            </SetZoom.Provider>
        </Zoom.Provider>
    );
};
