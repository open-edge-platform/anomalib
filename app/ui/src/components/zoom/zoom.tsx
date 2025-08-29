import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

type ZoomState = { scale: number; translate: { x: number; y: number } };
export const Zoom = createContext<ZoomState>({
    scale: 1.0,
    translate: { x: 0, y: 0 },
});
const SetZoom = createContext<Dispatch<SetStateAction<ZoomState>> | null>(null);

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

export const ZoomProvider = ({ children }: { children: ReactNode }) => {
    // TODO:
    // 1. Add scale restrictions - min max
    // 2. Add translate restrictions - min max
    const [zoom, setZoom] = useState<ZoomState>({
        scale: 1.0,
        translate: { x: 0, y: 0 },
    });

    return (
        <Zoom.Provider value={zoom}>
            <SetZoom.Provider value={setZoom}>{children}</SetZoom.Provider>
        </Zoom.Provider>
    );
};
