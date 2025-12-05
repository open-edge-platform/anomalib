import { ReactNode } from 'react';

import { Toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { NuqsAdapter } from 'nuqs/adapters/react-router/v6';
import { MemoryRouterProps, RouterProvider } from 'react-router';
import { MemoryRouter as Router } from 'react-router-dom';

import { WebRTCConnectionProvider } from './components/stream/web-rtc-connection-provider';
import { ZoomProvider } from './components/zoom/zoom';
import { queryClient } from './query-client/query-client';
import { router } from './routes/router';

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <WebRTCConnectionProvider>
                    <ZoomProvider>
                        <NuqsAdapter>
                            <RouterProvider router={router} />
                        </NuqsAdapter>
                    </ZoomProvider>
                </WebRTCConnectionProvider>
            </ThemeProvider>
        </QueryClientProvider>
    );
};

export const TestProviders = ({ children, routerProps }: { children: ReactNode; routerProps?: MemoryRouterProps }) => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <NuqsAdapter>
                    <Router {...routerProps}>
                        <WebRTCConnectionProvider>{children}</WebRTCConnectionProvider>
                        <Toast />
                    </Router>
                </NuqsAdapter>
            </ThemeProvider>
        </QueryClientProvider>
    );
};
