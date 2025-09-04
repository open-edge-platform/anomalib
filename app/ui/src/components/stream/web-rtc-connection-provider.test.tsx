import { fireEvent, render } from '@testing-library/react';
import { vi } from 'vitest';

import { Listener, WebRTCConnection, WebRTCConnectionStatus } from './web-rtc-connection';
import { useWebRTCConnection, WebRTCConnectionProvider } from './web-rtc-connection-provider';

vi.mock('./web-rtc-connection.ts');
class MockWebRTCConnection {
    status: WebRTCConnectionStatus = 'idle';
    listeners: Listener[] = [];

    public getStatus() {
        return this.status;
    }
    public getPeerConnection() {
        return undefined;
    }
    public getId() {
        return 'test';
    }

    public async start() {
        this.status = 'connected';
        this.listeners.forEach((l) => l({ type: 'status_change', status: this.status }));
    }
    public async stop() {
        this.status = 'idle';
        this.listeners.forEach((l) => l({ type: 'status_change', status: this.status }));
    }
    public subscribe(listener: Listener) {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter((currentListener) => currentListener !== listener);
        };
    }
}

describe('WebRTCConnectionProvider', () => {
    beforeEach(() => {
        // @ts-expect-error the mock implements all public methods
        vi.mocked(WebRTCConnection).mockImplementation(() => {
            return new MockWebRTCConnection();
        });
    });

    const App = () => {
        const { status, start, stop } = useWebRTCConnection();

        return (
            <>
                <span aria-label='status'>{status}</span>
                <button aria-label='start' onClick={start}>
                    Start
                </button>
                <button aria-label='stop' onClick={stop}>
                    Stop
                </button>
            </>
        );
    };

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('provides initial status as idle', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        expect(getByLabelText('status')).toHaveTextContent('idle');
    });

    it('updates status to connected after start', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connected');
    });

    it('updates status to idle after stop', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connected');

        fireEvent.click(getByLabelText('stop'));

        expect(getByLabelText('status')).toHaveTextContent('idle');
    });

    it('cleans up on unmount', () => {
        const stopSpy = vi.spyOn(MockWebRTCConnection.prototype, 'stop');

        const { unmount } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        unmount();

        expect(stopSpy).toHaveBeenCalled();
    });

    it('handles status sequence: start -> stop -> start', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connected');

        fireEvent.click(getByLabelText('stop'));
        expect(getByLabelText('status')).toHaveTextContent('idle');

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connected');
    });
});
