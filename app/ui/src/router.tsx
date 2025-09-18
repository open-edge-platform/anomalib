import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ErrorPage } from './components/error-page/error-page';
import { Layout } from './layout';
import { Inspect } from './routes/inspect/inspect';
import { OpenApi } from './routes/openapi/openapi';

const root = path('/');

export const paths = {
    root,
    openapi: root.path('/openapi'),
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Layout />
            </Suspense>
        ),
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                element: <Inspect />,
            },
            {
                path: paths.openapi.pattern,
                element: <OpenApi />,
            },
        ],
    },
]);
