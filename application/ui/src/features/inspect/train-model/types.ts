// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaModelFamily, SchemaTrainableModel } from '@geti-inspect/api/spec';

type ModelFamily = SchemaModelFamily;
export type TrainableModel = SchemaTrainableModel;

export const FAMILY_DISPLAY_NAMES: Record<ModelFamily, string> = {
    memory_bank: 'Memory Bank',
    distribution: 'Distribution',
    reconstruction: 'Reconstruction',
    student_teacher: 'Student-Teacher',
    gan_based: 'GAN-based',
    transformer: 'Transformer',
    foundation: 'Foundation',
    other: 'Other',
};
