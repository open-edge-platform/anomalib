import { fromOpenApi } from '@mswjs/source/open-api';
import { createOpenApiHttp, OpenApiHttpHandlers } from 'openapi-msw';

import { paths } from './openapi-spec';
import spec from './openapi-spec.json' with { type: 'json' };

const handlers = await fromOpenApi(JSON.stringify(spec).replace(/}:/g, '}//:'));

const getOpenApiHttp = (baseUrl?: string): OpenApiHttpHandlers<paths> => {
    const http = createOpenApiHttp<paths>({
        baseUrl: baseUrl ?? process.env.PUBLIC_API_BASE_URL ?? 'http://localhost:8000',
    });

    return {
        ...http,
        post: (path, ...other) => {
            // @ts-expect-error MSW internal parsing function does not accept paths like
            // `/api/models/{model_name}:activate`
            // to get around this we escape the colon character with `\\`
            // @see https://github.com/mswjs/msw/discussions/739
            return http.post(path.replace('}:', '}\\:'), ...other);
        },
    };
};

const http = getOpenApiHttp();

export { getOpenApiHttp, handlers, http };
