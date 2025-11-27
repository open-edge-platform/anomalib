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
import type { SchemaCompressionType, SchemaExportType } from 'src/api/openapi-spec';

import { downloadBlob, sanitizeFilename } from '../utils';
import type { ModelData } from './model-types';

const EXPORT_FORMATS: { id: SchemaExportType; name: string }[] = [
    { id: 'openvino', name: 'OpenVINO' },
    { id: 'onnx', name: 'ONNX' },
    { id: 'torch', name: 'PyTorch' },
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
    const [isExporting, setIsExporting] = useState(false);

    const handleFormatChange = (key: Key | null) => {
        if (key === null) return;
        const format = key as SchemaExportType;
        setSelectedFormat(format);

        if (format !== 'openvino') {
            setSelectedCompression('none');
        }
    };

    const handleCompressionChange = (key: Key | null) => {
        if (key === null) return;
        setSelectedCompression(key as SchemaCompressionType | 'none');
    };

    const handleExport = async () => {
        setIsExporting(true);

        try {
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
            downloadBlob(blob, filename);

            toast({ type: 'success', message: `Model "${model.name}" exported successfully.` });
            close();
        } catch {
            toast({ type: 'error', message: `Failed to export model "${model.name}".` });
        } finally {
            setIsExporting(false);
        }
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

                    <Picker
                        label='Export Format'
                        items={EXPORT_FORMATS}
                        selectedKey={selectedFormat}
                        onSelectionChange={handleFormatChange}
                        width='100%'
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>

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
                <Button variant='secondary' onPress={close} isDisabled={isExporting}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={handleExport} isPending={isExporting} isDisabled={isExporting}>
                    Export
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
