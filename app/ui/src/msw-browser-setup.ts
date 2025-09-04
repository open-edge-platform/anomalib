import { HttpResponse } from 'msw';
import { setupWorker } from 'msw/browser';

import { handlers, http } from './api/utils';

export const worker = setupWorker(
    ...handlers,
    http.get('/health', () => {
        return HttpResponse.json({ test: 'doei' });
    }),
    http.get('/api/system/metrics/memory', () => {
        return HttpResponse.json({});
    }),
    http.get('/api/models', () => {
        return HttpResponse.json([]);
    }),
    http.get('/api/sources', () => {
        return HttpResponse.json([]);
    }),
    http.get('/api/sinks', () => {
        return HttpResponse.json([]);
    }),
    http.get('/api/pipelines', () => {
        return HttpResponse.json([]);
    })
);
