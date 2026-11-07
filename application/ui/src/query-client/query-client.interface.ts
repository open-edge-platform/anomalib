/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * This file contains types for query keys and mutation metadata used in the API client.
 * It leverages the OpenAPI specification to ensure type safety and consistency.
 * The QueryKey type constructs query keys based on the API paths and their parameters.
 * The MutationMeta type defines metadata for mutations, including which queries to invalidate or await.
 * These types are used to enhance the functionality of the query client, particularly in managing cache invalidation
 * and ensuring data consistency after mutations.
 *
 * Be careful when modifying these types, as they are integral to the query client's operation.
 */

import type { HttpMethod } from 'openapi-typescript-helpers';

import { type paths } from '../api/openapi-spec';

type OperationFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> = Method extends keyof Paths[P]
    ? Paths[P][Method]
    : never;

type PathParamsFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> =
    OperationFor<Paths, P, Method> extends { parameters: { path: infer PP } } ? PP : never;

type MethodsForPath<Paths extends paths, P extends keyof Paths> = Extract<keyof Paths[P], HttpMethod>;

export type QueryKey<Paths extends paths> = {
    [P in keyof Paths]: {
        [M in MethodsForPath<Paths, P>]: PathParamsFor<Paths, P, M> extends never
            ? [M, P]
            : [
                  M,
                  P,
                  {
                      params: {
                          path: PathParamsFor<Paths, P, M>;
                      };
                  },
              ];
    }[MethodsForPath<Paths, P>];
}[keyof Paths];

export type MutationMeta = {
    invalidates?: QueryKey<paths>[];
    awaits?: QueryKey<paths>[];
};
