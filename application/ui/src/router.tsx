import { Suspense } from 'react';

import { IntelBrandedLoading } from '@geti/ui';
import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom';
import { path } from 'static-path';

import { $api } from './api/client';
import { ErrorPage } from './components/error-page/error-page';
import { Layout } from './layout';
import { Inspect } from './routes/inspect/inspect';
import { OpenApi } from './routes/openapi/openapi';

const root = path('/');
const projects = root.path('/projects');
const project = projects.path('/:projectId');

export const paths = {
    root,
    openapi: root.path('/openapi'),
    project,
};

const RedirectToProject = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/projects');

    const projectId = data.projects.at(0)?.id ?? '1';

    return <Navigate to={paths.project({ projectId })} replace />;
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        errorElement: <ErrorPage />,
        element: (
            <Suspense fallback={<IntelBrandedLoading />}>
                <Outlet />
            </Suspense>
        ),
        children: [
            {
                index: true,
                element: <RedirectToProject />,
            },
            {
                path: paths.project.pattern,
                element: <Layout />,
                children: [
                    {
                        index: true,
                        element: <Inspect />,
                    },
                ],
            },
            {
                path: paths.openapi.pattern,
                element: <OpenApi />,
            },
            {
                path: '*',
                element: <RedirectToProject />,
            },
        ],
    },
]);
