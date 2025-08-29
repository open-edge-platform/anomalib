import { createContext, ReactNode, RefObject, useCallback, useContext, useEffect, useRef, useState } from 'react';

import { WebRTCConnection, WebRTCConnectionStatus } from './web-rtc-connection';

export type WebRTCConnectionState = null | {
    status: WebRTCConnectionStatus;
    start: () => Promise<void>;
    stop: () => Promise<void>;
    webRTCConnectionRef: RefObject<WebRTCConnection>;
};

export const WebRTCConnectionContext = createContext<WebRTCConnectionState>(null);

const useWebRTCConnectionState = () => {
    const webRTCConnectionRef = useRef<WebRTCConnection | null>(null);
    const [status, setStatus] = useState<WebRTCConnectionStatus>('idle');

    // Initialize WebRTCConnection on mount
    useEffect(() => {
        if (webRTCConnectionRef.current) {
            return;
        }

        const webRTCConnection = new WebRTCConnection();
        webRTCConnectionRef.current = webRTCConnection;

        const unsubscribe = webRTCConnection.subscribe((event) => {
            if (event.type === 'status_change') {
                setStatus(event.status);
            }

            if (event.type === 'error') {
                console.error('WebRTC Connection Error:', event.error);
                // Optionally update status to 'failed' if not already
                if (webRTCConnectionRef.current?.getStatus() !== 'failed') {
                    setStatus('failed');
                }
            }
        });

        return () => {
            unsubscribe();
            webRTCConnection.stop(); // Ensure connection is closed on unmount
            webRTCConnectionRef.current = null;
        };
    }, []);

    const start = useCallback(async () => {
        if (!webRTCConnectionRef.current) {
            return;
        }

        try {
            await webRTCConnectionRef.current.start();
        } catch (error) {
            console.error('Failed to start WebRTC connection:', error);
            setStatus('failed');
        }
    }, []);

    const stop = useCallback(async () => {
        if (!webRTCConnectionRef.current) {
            return;
        }

        await webRTCConnectionRef.current.stop();
    }, []);

    return {
        start,
        stop,
        status,
        webRTCConnectionRef,
    };
};

export const WebRTCConnectionProvider = ({ children }: { children: ReactNode }) => {
    const value = useWebRTCConnectionState();

    return <WebRTCConnectionContext.Provider value={value}>{children}</WebRTCConnectionContext.Provider>;
};

export const useWebRTCConnection = () => {
    const context = useContext(WebRTCConnectionContext);

    if (context === null) {
        throw new Error('useWebRTCConnection was used outside of WebRTCConnectionProvider');
    }

    return context;
};
