import { $api } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import { Item, Key, Picker, toast } from '@geti/ui';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

export const InferenceDevices = () => {
    const { data: pipeline } = usePipeline();
    const { data } = $api.useSuspenseQuery('get', '/api/inference-devices');
    const { projectId } = useProjectIdentifier();
    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        onError: (error) => {
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
    });

    const devices = data.devices;
    const options = devices.map((device) => ({ id: device, name: device }));
    const defaultSelectedKey = pipeline.inference_device?.toLowerCase() ?? undefined;

    const handleChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        updatePipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { inference_device: key },
        });
    };

    return (
        <Picker
            width='100%'
            items={options}
            label='Inference devices'
            onSelectionChange={handleChange}
            defaultSelectedKey={defaultSelectedKey}
        >
            {(item) => <Item>{item.name}</Item>}
        </Picker>
    );
};
