import React from 'react';

import ReactDOM from 'react-dom/client';

import { Providers } from './providers';

import './index.css';

async function enableMocking() {
    if (process.env.NODE_ENV !== 'development') {
        return;
    }

    const { worker } = await import('./msw-browser-setup');

    // `worker.start()` returns a Promise that resolves
    // once the Service Worker is up and ready to intercept requests.
    return worker.start();
}

enableMocking().then(() => {
    const rootEl = document.getElementById('root');
    if (rootEl) {
        const root = ReactDOM.createRoot(rootEl);
        root.render(
            <React.StrictMode>
                <Providers />
            </React.StrictMode>
        );
    }
});
