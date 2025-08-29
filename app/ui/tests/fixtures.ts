import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';

import { handlers, http } from '../src/api/utils';

interface Fixtures {
    network: NetworkFixture;
}

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: [
            ...handlers,
            http.get('/api/system/metrics/memory', ({ response }) => {
                return response(200).json({});
            }),
            http.get('/api/models', ({ response }) => {
                return response(200).json([]);
            }),
            http.post('/api/webrtc/offer', ({ response }) => {
                // Schema is empty, so we return an empty object
                return response(200).json({} as never);
            }),
            http.post('/api/input_hook', ({ response }) => {
                // Schema is empty, so we return an empty object
                return response(200).json({} as never);
            }),
        ],
    }),
});

export { expect, http, test };
