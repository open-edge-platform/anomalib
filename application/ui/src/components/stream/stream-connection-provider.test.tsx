import { fireEvent, render } from '@testing-library/react';

import { StreamConnectionProvider, useStreamConnection } from './stream-connection-provider';

describe('StreamConnectionProvider', () => {
    const App = () => {
        const { status, start, stop, streamUrl, setStatus } = useStreamConnection();

        return (
            <>
                <span aria-label='status'>{status}</span>
                <span aria-label='stream-url'>{streamUrl}</span>
                <button aria-label='start' onClick={start}>
                    Start
                </button>
                <button aria-label='stop' onClick={stop}>
                    Stop
                </button>
                <button aria-label='connected' onClick={() => setStatus('connected')}>
                    Connected
                </button>
            </>
        );
    };

    it('provides initial status as idle', () => {
        const { getByLabelText } = render(
            <StreamConnectionProvider>
                <App />
            </StreamConnectionProvider>
        );

        expect(getByLabelText('status')).toHaveTextContent('idle');
        expect(getByLabelText('stream-url')).toHaveTextContent('');
    });

    it('updates status to connecting after start', () => {
        const { getByLabelText } = render(
            <StreamConnectionProvider>
                <App />
            </StreamConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connecting');
        expect(getByLabelText('stream-url')).not.toHaveTextContent('');
    });

    it('updates status to idle after stop', () => {
        const { getByLabelText } = render(
            <StreamConnectionProvider>
                <App />
            </StreamConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connecting');

        fireEvent.click(getByLabelText('stop'));

        expect(getByLabelText('status')).toHaveTextContent('idle');
        expect(getByLabelText('stream-url')).toHaveTextContent('');
    });

    it('handles status sequence: start -> connected -> stop -> start', () => {
        const { getByLabelText } = render(
            <StreamConnectionProvider>
                <App />
            </StreamConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connecting');

        fireEvent.click(getByLabelText('connected'));
        expect(getByLabelText('status')).toHaveTextContent('connected');

        fireEvent.click(getByLabelText('stop'));
        expect(getByLabelText('status')).toHaveTextContent('idle');

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connecting');
    });
});
