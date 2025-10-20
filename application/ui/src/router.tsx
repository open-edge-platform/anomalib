import { Suspense } from 'react';

import { IntelBrandedLoading } from '@geti/ui';
import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom';
import { path } from 'static-path';

import { $api } from './api/client';
import { ErrorPage } from './components/error-page/error-page';
import { Layout } from './layout';
import { Inspect } from './routes/inspect/inspect';
import { OpenApi } from './routes/openapi/openapi';
import { Welcome } from './routes/welcome';

const root = path('/');
const projects = root.path('/projects');
const project = projects.path('/:projectId');
const welcome = path('/welcome');

export const paths = {
    root,
    openapi: root.path('/openapi'),
    project,
    welcome,
};

const Redirect = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/projects');
    const projects = data.projects;

    if (projects.length === 0) {
        return <Navigate to={paths.welcome({})} replace />;
    }

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
                element: <Redirect />,
            },
            {
                path: paths.welcome.pattern,
                element: <Welcome />,
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
                element: <Redirect />,
            },
        ],
    },
]);
