import { useState } from 'react';

import { $api, fetchClient } from '@geti-inspect/api';
import { useProjectIdentifier } from '@geti-inspect/hooks';
import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Item,
    Picker,
    Text,
    toast,
    type Key,
} from '@geti/ui';
import { useMutation } from '@tanstack/react-query';
import type { SchemaCompressionType, SchemaExportType } from 'src/api/openapi-spec';
import { Onnx, OpenVino, PyTorch } from 'src/assets/icons';

import type { ModelData } from '../../../hooks/utils';
import { downloadBlob, sanitizeFilename } from '../utils';

import classes from './export-model-dialog.module.scss';

const EXPORT_FORMATS: { id: SchemaExportType; name: string; Icon: React.FC<React.SVGProps<SVGSVGElement>> }[] = [
    { id: 'openvino', name: 'OpenVINO', Icon: OpenVino },
    { id: 'onnx', name: 'ONNX', Icon: Onnx },
    { id: 'torch', name: 'PyTorch', Icon: PyTorch },
];

const COMPRESSION_OPTIONS: { id: SchemaCompressionType | 'none'; name: string }[] = [
    { id: 'none', name: 'None' },
    { id: 'fp16', name: 'FP16' },
    { id: 'int8', name: 'INT8' },
    { id: 'int8_ptq', name: 'INT8 PTQ' },
    { id: 'int8_acq', name: 'INT8 ACQ' },
];

interface ExportModelDialogProps {
    model: ModelData;
    close: () => void;
}

export const ExportModelDialog = ({ model, close }: ExportModelDialogProps) => {
    const { projectId } = useProjectIdentifier();
    const { data: project } = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });
    const [selectedFormat, setSelectedFormat] = useState<SchemaExportType>('openvino');
    const [selectedCompression, setSelectedCompression] = useState<SchemaCompressionType | 'none'>('none');

    const exportMutation = useMutation({
        mutationFn: async () => {
            const compression = selectedCompression === 'none' ? null : selectedCompression;

            const response = await fetchClient.POST('/api/projects/{project_id}/models/{model_id}:export', {
                params: {
                    path: {
                        project_id: projectId,
                        model_id: model.id,
                    },
                },
                body: {
                    format: selectedFormat,
                    compression,
                },
                parseAs: 'blob',
            });

            if (response.error) {
                throw new Error('Export failed');
            }

            const blob = response.data as Blob;
            const compressionSuffix = compression ? `_${compression}` : '';
            const sanitizedProjectName = sanitizeFilename(project.name);
            const sanitizedModelName = sanitizeFilename(model.name);
            const filename = `${sanitizedProjectName}_${sanitizedModelName}_${selectedFormat}${compressionSuffix}.zip`;

            return { blob, filename };
        },
        onSuccess: ({ blob, filename }) => {
            downloadBlob(blob, filename);
            toast({ type: 'success', message: `Model "${model.name}" exported successfully.` });
            close();
        },
        onError: () => {
            toast({ type: 'error', message: `Failed to export model "${model.name}".` });
        },
    });

    const handleFormatChange = (value: string) => {
        const format = value as SchemaExportType;
        setSelectedFormat(format);

        if (format !== 'openvino') {
            setSelectedCompression('none');
        }
    };

    const handleCompressionChange = (key: Key | null) => {
        if (key === null) return;
        setSelectedCompression(key as SchemaCompressionType | 'none');
    };

    return (
        <Dialog size='S'>
            <Heading>Export Model</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        Export <strong>{model.name}</strong> to a downloadable format.
                    </Text>

                    <Flex direction='column' gap='size-100'>
                        <Text UNSAFE_className={classes.label}>Export Format</Text>
                        <div className={classes.formatGroup} role='radiogroup' aria-label='Select export format'>
                            {EXPORT_FORMATS.map(({ id, Icon }) => (
                                <button
                                    key={id}
                                    type='button'
                                    role='radio'
                                    aria-checked={selectedFormat === id}
                                    onClick={() => handleFormatChange(id)}
                                    className={`${classes.formatOption} ${
                                        selectedFormat === id ? classes.formatOptionSelected : ''
                                    }`}
                                >
                                    <Icon className={classes.formatIcon} />
                                </button>
                            ))}
                        </div>
                    </Flex>

                    {selectedFormat === 'openvino' && (
                        <Picker
                            label='Compression (optional)'
                            items={COMPRESSION_OPTIONS}
                            selectedKey={selectedCompression}
                            onSelectionChange={handleCompressionChange}
                            width='100%'
                        >
                            {(item) => <Item key={item.id}>{item.name}</Item>}
                        </Picker>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close} isDisabled={exportMutation.isPending}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={() => exportMutation.mutate()}
                    isPending={exportMutation.isPending}
                    isDisabled={exportMutation.isPending}
                >
                    Export
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
