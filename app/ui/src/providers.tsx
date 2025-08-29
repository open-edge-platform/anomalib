import { ReactNode } from 'react';

import { ThemeProvider } from '@geti/ui/theme';
import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouterProps, RouterProvider } from 'react-router';
import { MemoryRouter as Router } from 'react-router-dom';

import { WebRTCConnectionProvider } from './components/stream/web-rtc-connection-provider';
import { ZoomProvider } from './components/zoom/zoom';
import { router } from './router';

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            gcTime: 30 * 60 * 1000,
            staleTime: 5 * 60 * 1000,
            networkMode: 'always',
        },
        mutations: {
            networkMode: 'always',
        },
    },
    mutationCache: new MutationCache({
        onSuccess: () => {
            queryClient.invalidateQueries();
        },
    }),
});

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <WebRTCConnectionProvider>
                    <ZoomProvider>
                        <RouterProvider router={router} />
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
