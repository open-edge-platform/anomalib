import '@testing-library/jest-dom';

import fetchPolyfill, { Request as RequestPolyfill } from 'node-fetch';
import { afterAll, afterEach, beforeAll } from 'vitest';

import { server } from './msw-node-setup';

process.env.PUBLIC_API_BASE_URL = 'http://localhost:8000';

beforeAll(() => {
    server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => {
    server.close();
});

// Why we need these polyfills:
// https://github.com/reduxjs/redux-toolkit/issues/4966#issuecomment-3115230061
Object.defineProperty(global, 'fetch', {
    // MSW will overwrite this to intercept requests
    writable: true,
    value: fetchPolyfill,
});

Object.defineProperty(global, 'Request', {
    writable: false,
    value: RequestPolyfill,
});

window.ResizeObserver = class ResizeObserver {
    observe() {
        // empty
    }
    unobserve() {
        // empty
    }
    disconnect() {
        // empty
    }
};

class MockIntersectionObserver {
    root = null;
    rootMargin = '0px';
    thresholds = [];

    constructor() {}

    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords(): IntersectionObserverEntry[] {
        return [];
    }
}

window.IntersectionObserver = MockIntersectionObserver;
