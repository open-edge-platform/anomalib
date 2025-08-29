import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { Layout } from './layout';
import { Inference } from './routes/inference/inference';

const root = path('/');
const inference = root.path('/inference');

export const paths = {
    root,
    inference: {
        index: inference,
    },
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Layout />
            </Suspense>
        ),
        errorElement: <div>Oh no</div>,
        children: [
            {
                index: true,
                loader: () => {
                    // TODO: if no pipeline configured then redirect to source
                    // else redirect to live-feed
                    return redirect(paths.inference.index({}));
                },
            },
            {
                path: paths.inference.index.pattern,
                element: <Inference />,
            },
        ],
    },
]);
