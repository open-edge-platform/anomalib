import { setupWorker } from 'msw/browser';

import { handlers } from './api/utils';

export const worker = setupWorker(...handlers);
