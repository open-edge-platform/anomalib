import { setupServer } from 'msw/node';

import { handlers } from './api/utils';

// Initialize msw's mock server with the handlers
export const server = setupServer(...handlers);
