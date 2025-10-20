import { ReactNode } from 'react';

import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { Toast } from 'packages/ui';
import { MemoryRouterProps, RouterProvider } from 'react-router';
import { MemoryRouter as Router } from 'react-router-dom';

import { WebRTCConnectionProvider } from './components/stream/web-rtc-connection-provider';
import { ZoomProvider } from './components/zoom/zoom';
import { queryClient } from './query-client/query-client';
import { router } from './router';

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <WebRTCConnectionProvider>
                    <ZoomProvider>
                        <RouterProvider router={router} />
                        <Toast />
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
                <Router {...routerProps}>
                    <WebRTCConnectionProvider>{children}</WebRTCConnectionProvider>
                </Router>
            </ThemeProvider>
        </QueryClientProvider>
    );
};
