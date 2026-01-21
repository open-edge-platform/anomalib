// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fetchClient } from '@geti-inspect/api';
import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Item,
    Picker,
    Text,
    TextArea,
    Tooltip,
    TooltipTrigger,
    View,
} from '@geti/ui';
import { DownloadIcon, ExternalLinkIcon, HelpIcon } from '@geti/ui/icons';
import { useMutation } from '@tanstack/react-query';

import { downloadBlob } from '../utils';

const GITHUB_ISSUES_URL = 'https://github.com/open-edge-platform/anomalib/issues/new';

interface LibraryVersions {
    python: string;
    pytorch: string | null;
    lightning: string | null;
    torchmetrics: string | null;
    openvino: string | null;
    onnx: string | null;
    anomalib: string | null;
}

interface AcceleratorInfo {
    cuda_available: boolean;
    cuda_version: string | null;
    cudnn_version: string | null;
    gpu_name: string | null;
    xpu_available: boolean;
    xpu_version: string | null;
    xpu_name: string | null;
}

interface SystemInfo {
    os_name: string;
    os_version: string;
    platform: string;
    app_version: string;
    app_name: string;
    libraries: LibraryVersions;
    accelerators: AcceleratorInfo;
}

type IssueType = 'bug' | 'feature';

const ISSUE_TYPE_OPTIONS = [
    { id: 'bug', label: 'Bug Report' },
    { id: 'feature', label: 'Feature Request' },
] as const;

const fetchSystemInfo = async (): Promise<SystemInfo> => {
    const response = await fetchClient.GET('/api/system/info', {});
    return response.data as SystemInfo;
};

const formatLibraryVersion = (name: string, version: string | null): string => {
    return version ? `${name}: ${version}` : `${name}: not installed`;
};

const formatAcceleratorInfo = (accelerators: AcceleratorInfo): string[] => {
    const lines: string[] = [];

    if (accelerators.cuda_available) {
        lines.push(`CUDA: ${accelerators.cuda_version || 'available'}`);
        if (accelerators.cudnn_version) {
            lines.push(`cuDNN: ${accelerators.cudnn_version}`);
        }
        if (accelerators.gpu_name) {
            lines.push(`GPU: ${accelerators.gpu_name}`);
        }
    }

    if (accelerators.xpu_available) {
        lines.push(`Intel XPU: ${accelerators.xpu_version || 'available'}`);
        if (accelerators.xpu_name) {
            lines.push(`Device: ${accelerators.xpu_name}`);
        }
    }

    if (lines.length === 0) {
        lines.push('No GPU acceleration detected (CPU only)');
    }

    return lines;
};

const createGitHubIssueUrl = (systemInfo: SystemInfo, issueType: IssueType, description: string): string => {
    const isBug = issueType === 'bug';
    const title = isBug ? '[Bug]: ' : '[Feature]: ';
    const labels = isBug ? ['bug', 'Geti Inspect'] : ['enhancement', 'Geti Inspect'];

    const { libraries, accelerators } = systemInfo;

    const libraryLines = [
        formatLibraryVersion('Python', libraries.python),
        formatLibraryVersion('PyTorch', libraries.pytorch),
        formatLibraryVersion('Lightning', libraries.lightning),
        formatLibraryVersion('TorchMetrics', libraries.torchmetrics),
        formatLibraryVersion('OpenVINO', libraries.openvino),
        formatLibraryVersion('ONNX', libraries.onnx),
        formatLibraryVersion('Anomalib', libraries.anomalib),
    ];

    const acceleratorLines = formatAcceleratorInfo(accelerators);

    const body = `## System Information

### Environment
- **OS**: ${systemInfo.os_name} ${systemInfo.os_version}
- **Platform**: ${systemInfo.platform}
- **App**: ${systemInfo.app_name} v${systemInfo.app_version}

### Library Versions
${libraryLines.map((line) => `- ${line}`).join('\n')}

### Hardware Acceleration
${acceleratorLines.map((line) => `- ${line}`).join('\n')}

## ${isBug ? 'Bug Description' : 'Feature Description'}
${description || '_Please describe the issue or feature request_'}
`;

    const params = new URLSearchParams({
        title,
        body,
        labels: labels.join(','),
    });

    return `${GITHUB_ISSUES_URL}?${params.toString()}`;
};

const downloadLogs = async (): Promise<void> => {
    const response = await fetchClient.GET('/api/system/logs', {
        parseAs: 'blob',
    });

    if (response.data) {
        const contentDisposition = response.response.headers.get('content-disposition');
        const filenameMatch = contentDisposition?.match(/filename=(.+)/);
        const filename = filenameMatch ? filenameMatch[1] : 'geti_inspect_logs.zip';

        downloadBlob(response.data as Blob, filename);
    }
};

interface SubmitFeedbackParams {
    issueType: IssueType;
    description: string;
}

const submitFeedback = async ({ issueType, description }: SubmitFeedbackParams): Promise<void> => {
    const systemInfo = await fetchSystemInfo();
    const issueUrl = createGitHubIssueUrl(systemInfo, issueType, description);
    window.open(issueUrl, '_blank', 'noopener,noreferrer');
};

interface FeedbackDialogContentProps {
    close: () => void;
}

const FeedbackDialogContent = ({ close }: FeedbackDialogContentProps) => {
    const [issueType, setIssueType] = useState<IssueType>('bug');
    const [description, setDescription] = useState('');

    const submitMutation = useMutation({
        mutationFn: submitFeedback,
        onSuccess: () => close(),
    });

    const downloadLogsMutation = useMutation({
        mutationFn: downloadLogs,
    });

    const error = submitMutation.error || downloadLogsMutation.error;

    return (
        <>
            <Heading>Submit Feedback</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        Help us improve by reporting bugs or suggesting new features. Your feedback will be submitted as
                        a GitHub issue with system information automatically included.
                    </Text>

                    <Picker
                        label='Issue Type'
                        selectedKey={issueType}
                        onSelectionChange={(key) => setIssueType(key as IssueType)}
                        width='100%'
                    >
                        {ISSUE_TYPE_OPTIONS.map((option) => (
                            <Item key={option.id}>{option.label}</Item>
                        ))}
                    </Picker>

                    <TextArea
                        label='Description'
                        placeholder={
                            issueType === 'bug'
                                ? 'Describe what happened and what you expected to happen...'
                                : 'Describe the feature you would like to see...'
                        }
                        value={description}
                        onChange={setDescription}
                        width='100%'
                        height='size-1600'
                    />

                    {issueType === 'bug' && (
                        <View backgroundColor='gray-100' padding='size-150' borderRadius='regular'>
                            <Flex direction='row' alignItems='center' justifyContent='space-between' gap='size-100'>
                                <Text
                                    UNSAFE_style={{ fontSize: '12px', color: 'var(--spectrum-global-color-gray-700)' }}
                                >
                                    Optionally download and attach application logs to help diagnose the issue.
                                </Text>
                                <Button
                                    variant='secondary'
                                    onPress={() => downloadLogsMutation.mutate()}
                                    isPending={downloadLogsMutation.isPending}
                                    isDisabled={submitMutation.isPending}
                                >
                                    <DownloadIcon size='S' />
                                    <Text>Download Logs</Text>
                                </Button>
                            </Flex>
                        </View>
                    )}

                    {error && (
                        <View
                            backgroundColor='negative'
                            padding='size-100'
                            borderRadius='regular'
                            UNSAFE_style={{ color: 'var(--spectrum-global-color-red-700)' }}
                        >
                            <Text>
                                {error instanceof Error
                                    ? error.message
                                    : 'An error occurred. Please try again.'}
                            </Text>
                        </View>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={() => submitMutation.mutate({ issueType, description })}
                    isPending={submitMutation.isPending}
                >
                    <ExternalLinkIcon size='S' />
                    <Text>Open GitHub Issue</Text>
                </Button>
            </ButtonGroup>
        </>
    );
};

export const FeedbackButton = () => {
    return (
        <DialogTrigger type='modal'>
            <TooltipTrigger delay={0}>
                <ActionButton isQuiet aria-label='Submit feedback'>
                    <HelpIcon />
                </ActionButton>
                <Tooltip>Submit Feedback</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog>
                    <FeedbackDialogContent close={close} />
                </Dialog>
            )}
        </DialogTrigger>
    );
};
