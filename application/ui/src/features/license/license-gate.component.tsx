// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { IntelBrandedLoading, AlertDialog, DialogContainer, Flex, Link, Text } from '@geti/ui';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { getApiUrl } from '../../api/client';

type LicenseStatus = {
    accepted: boolean;
    accepted_version: string | null;
    app_version: string;
    deployment_type: 'win_app' | 'docker' | 'dev';
    licenses: Array<{
        name: string;
        url: string;
        required_for: string;
    }>;
};

interface LicenseGateProps {
    children: ReactNode;
}

const DeploymentLabel: Record<LicenseStatus['deployment_type'], string> = {
    win_app: 'Windows application',
    docker: 'Docker deployment',
    dev: 'development deployment',
};

const fetchLicenseStatus = async (): Promise<LicenseStatus> => {
    const response = await fetch(getApiUrl('/api/system/license'));

    if (!response.ok) {
        throw new Error('Failed to load license status.');
    }

    return (await response.json()) as LicenseStatus;
};

const acceptLicenses = async (): Promise<void> => {
    const response = await fetch(getApiUrl('/api/system/license:accept'), { method: 'POST' });

    if (!response.ok) {
        throw new Error('Failed to accept licenses.');
    }
};

export const LicenseGate = ({ children }: LicenseGateProps) => {
    const [isOpen, setIsOpen] = useState(true);
    const queryClient = useQueryClient();

    const { data: licenseStatus, isPending, isError } = useQuery({
        queryKey: ['license-status'],
        queryFn: fetchLicenseStatus,
        retry: false,
    });
    const acceptMutation = useMutation({
        mutationFn: acceptLicenses,
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: ['license-status'] });
            setIsOpen(false);
        },
    });

    if (isPending || licenseStatus === undefined) {
        return <IntelBrandedLoading />;
    }

    if (isError) {
        return (
            <Flex alignItems='center' justifyContent='center' height='100vh'>
                <Text>Failed to load required license information.</Text>
            </Flex>
        );
    }

    const shouldBlock = isOpen && !licenseStatus.accepted;

    return (
        <>
            {children}
            <DialogContainer onDismiss={() => undefined}>
                {shouldBlock ? (
                    <AlertDialog
                        variant='confirmation'
                        cancelLabel='Close'
                        title={`Review licenses for Anomalib Studio ${licenseStatus.app_version}`}
                        primaryActionLabel={acceptMutation.isPending ? 'Accepting...' : 'Accept and continue'}
                        isPrimaryActionDisabled={acceptMutation.isPending}
                        onPrimaryAction={() => acceptMutation.mutate()}
                    >
                        <Flex direction='column' gap='size-200'>
                            <Text>
                                Please review and accept the licenses required for this{' '}
                                {DeploymentLabel[licenseStatus.deployment_type]}. The dialog reappears whenever the Studio
                                version changes.
                            </Text>

                            <Flex direction='column' gap='size-150'>
                                {licenseStatus.licenses.map((license) => (
                                    <Flex key={`${license.required_for}-${license.name}`} direction='column' gap='size-50'>
                                        <Text>{license.required_for}</Text>
                                        <Link href={license.url} target='_blank' rel='noreferrer'>
                                            {license.name}
                                        </Link>
                                    </Flex>
                                ))}
                            </Flex>

                            {licenseStatus.accepted_version ? (
                                <Text>Previously accepted version: {licenseStatus.accepted_version}</Text>
                            ) : null}

                            {acceptMutation.isError ? (
                                <Text>Failed to store license acceptance. Try again.</Text>
                            ) : null}
                        </Flex>
                    </AlertDialog>
                ) : null}
            </DialogContainer>
        </>
    );
};
