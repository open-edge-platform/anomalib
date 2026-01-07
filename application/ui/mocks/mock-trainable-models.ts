// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaTrainableModel } from '../src/api/openapi-spec';

export const getMockedTrainableModel = (partial: Partial<SchemaTrainableModel> = {}): SchemaTrainableModel => ({
    id: 'patchcore',
    name: 'PatchCore',
    family: ['memory_bank'],
    description: 'A memory bank based anomaly detection model.',
    license: 'Apache-2.0',
    docs_url: 'https://anomalib.readthedocs.io/',
    recommended: true,
    parameters: 27,
    ...partial,
});

export const TRAINABLE_MODELS: SchemaTrainableModel[] = [
    getMockedTrainableModel({ id: 'patchcore', name: 'PatchCore', recommended: true }),
    getMockedTrainableModel({ id: 'dinomaly', name: 'Dinomaly', recommended: true }),
    getMockedTrainableModel({ id: 'padim', name: 'PaDiM', recommended: false }),
    getMockedTrainableModel({ id: 'stfpm', name: 'STFPM', recommended: false }),
];
