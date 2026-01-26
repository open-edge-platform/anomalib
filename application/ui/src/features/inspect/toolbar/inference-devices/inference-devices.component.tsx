import { useState } from 'react';

import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Item, Key, Picker, toast } from '@geti/ui';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

export const InferenceDevices = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/system/devices/inference');
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const [selectedKey, setSelectedKey] = useState<Key | null>(pipeline.inference_device?.toLowerCase() ?? null);

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
        onError: (error) => {
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
    });

    const options = data.map((device) => {
        const id = device.type.toLowerCase();
        return { id, name: device.type };
    });

    const handleChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        setSelectedKey(key);
        updatePipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { inference_device: key },
        });
    };

    return (
        <Picker
            maxWidth={'size-3000'}
            label='Inference devices: '
            aria-label='inference devices'
            labelAlign='end'
            labelPosition='side'
            items={options}
            onSelectionChange={handleChange}
            selectedKey={selectedKey}
        >
            {(item) => <Item>{item.name}</Item>}
        </Picker>
    );
};
