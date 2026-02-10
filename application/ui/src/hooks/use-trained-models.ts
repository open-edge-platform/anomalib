// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';

export const useTrainedModels = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data.models;
};
