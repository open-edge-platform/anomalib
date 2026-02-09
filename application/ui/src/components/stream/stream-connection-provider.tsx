import { createContext, Dispatch, ReactNode, SetStateAction, useCallback, useContext, useState } from 'react';

export type StreamConnectionStatus = 'idle' | 'connecting' | 'connected' | 'failed' | 'disconnected';

export type StreamConnectionState = null | {
    status: StreamConnectionStatus;
    start: () => Promise<void>;
    stop: () => Promise<void>;
    streamUrl: string | null;
    setStatus: Dispatch<SetStateAction<StreamConnectionStatus>>;
};

const STREAM_ENDPOINT = '/api/stream';

export const StreamConnectionContext = createContext<StreamConnectionState>(null);

const useStreamConnectionState = () => {
    const [status, setStatus] = useState<StreamConnectionStatus>('idle');
    const [streamUrl, setStreamUrl] = useState<string | null>(null);

    const start = useCallback(async () => {
        setStatus('connecting');
        setStreamUrl(`${STREAM_ENDPOINT}?ts=${Date.now()}`);
    }, []);

    const stop = useCallback(async () => {
        setStreamUrl(null);
        setStatus('idle');
    }, []);

    return {
        status,
        start,
        stop,
        streamUrl,
        setStatus,
    };
};

export const StreamConnectionProvider = ({ children }: { children: ReactNode }) => {
    const value = useStreamConnectionState();

    return <StreamConnectionContext.Provider value={value}>{children}</StreamConnectionContext.Provider>;
};

export const useStreamConnection = () => {
    const context = useContext(StreamConnectionContext);

    if (context === null) {
        throw new Error('useStreamConnection was used outside of StreamConnectionProvider');
    }

    return context;
};
