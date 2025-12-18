// Connect to an SSE endpoint and yield its messages
export function fetchSSE<T = unknown>(url: string) {
    return {
        async *[Symbol.asyncIterator](): AsyncGenerator<T> {
            const eventSource = new EventSource(url);

            try {
                let { promise, resolve, reject } = Promise.withResolvers<string>();

                eventSource.onmessage = (event) => {
                    if (event.data === 'DONE' || event.data.includes('COMPLETED')) {
                        eventSource.close();
                        resolve('DONE');
                        return;
                    }
                    resolve(event.data);
                };

                eventSource.onerror = (error) => {
                    eventSource.close();
                    reject(new Error('EventSource failed: ' + error));
                };

                // Keep yielding data as it comes in
                while (true) {
                    const message = await promise;

                    // If server sends 'DONE' message or similar, break the loop
                    if (message === 'DONE') {
                        break;
                    }

                    try {
                        yield JSON.parse(message);
                    } catch {
                        console.error('Could not parse message:', message);
                    }

                    ({ promise, resolve, reject } = Promise.withResolvers<string>());
                }
            } finally {
                eventSource.close();
            }
        },
    };
}
