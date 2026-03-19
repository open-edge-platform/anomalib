import { Suspense } from 'react';

import { useStartupProjectSelection } from '@anomalib-studio/hooks';
import { IntelBrandedLoading, Toast } from '@geti/ui';
import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom';

import { ErrorPage } from './../components/error-page/error-page';
import { Inspect } from './inspect/inspect';
import { Layout } from './layout';
import { OpenApi } from './openapi/openapi';
import { getInspectProjectPath, paths } from './paths';
import { Welcome } from './welcome';

const Redirect = () => {
    const { data } = useStartupProjectSelection();

    if (data.project_id === null) {
        return <Navigate to={paths.welcome({})} replace />;
    }

    return <Navigate to={getInspectProjectPath(data.project_id)} replace />;
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        errorElement: <ErrorPage />,
        element: (
            <Suspense fallback={<IntelBrandedLoading />}>
                <Toast />

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
