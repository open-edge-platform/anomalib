// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Item, Picker, Text, type Key } from '@geti/ui';
import type { SchemaCompressionType, SchemaExportType } from 'src/api/openapi-spec';
import { Onnx, OpenVino, PyTorch } from 'src/assets/icons';

import type { ModelData } from '../../../hooks/utils';

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

export interface ExportOptions {
    format: SchemaExportType;
    formatLabel: string;
    compression: SchemaCompressionType | null;
}

interface ExportModelDialogProps {
    model: ModelData;
    close: () => void;
    onExport: (options: ExportOptions) => void;
}

export const ExportModelDialog = ({ model, close, onExport }: ExportModelDialogProps) => {
    const [selectedFormat, setSelectedFormat] = useState<SchemaExportType>('openvino');
    const [selectedCompression, setSelectedCompression] = useState<SchemaCompressionType | 'none'>('none');

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

    const handleExport = () => {
        const formatLabel = EXPORT_FORMATS.find((f) => f.id === selectedFormat)?.name ?? selectedFormat;
        const compression = selectedCompression === 'none' ? null : selectedCompression;

        onExport({ format: selectedFormat, formatLabel, compression });
        close();
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
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={handleExport}>
                    Export
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
