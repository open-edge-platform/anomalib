// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { AlertDialog, DialogContainer, Flex, IntelBrandedLoading, Link, Text } from '@geti/ui';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { getApiUrl } from '../../api/client';

type LicenseStatus = {
    accepted: boolean;
    app_version: string;
    deployment_type: 'win_app' | 'docker' | 'dev';
    license: {
        name: string;
        url: string;
        required_for: string;
    } | null;
};

interface LicenseGateProps {
    children: ReactNode;
}

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
    const queryClient = useQueryClient();

    const {
        data: licenseStatus,
        isPending,
        isError,
    } = useQuery({
        queryKey: ['license-status'],
        queryFn: fetchLicenseStatus,
        retry: false,
    });
    const acceptMutation = useMutation({
        mutationFn: acceptLicenses,
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: ['license-status'] });
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

    const shouldBlock = !licenseStatus.accepted;

    return (
        <>
            {children}
            <DialogContainer onDismiss={() => undefined}>
                {shouldBlock ? (
                    <AlertDialog
                        variant='confirmation'
                        title={`License Agreement — Anomalib Studio ${licenseStatus.app_version}`}
                        cancelLabel={undefined}
                        primaryActionLabel={acceptMutation.isPending ? 'Accepting...' : 'Accept and continue'}
                        isPrimaryActionDisabled={acceptMutation.isPending}
                        onPrimaryAction={() => acceptMutation.mutate()}
                    >
                        <Flex direction='column' gap='size-200'>
                            <Text>
                                By continuing you agree to the{' '}
                                <Link
                                    href={licenseStatus.license?.url ?? '#'}
                                    target='_blank'
                                    rel='noreferrer'
                                >
                                    {licenseStatus.license?.name ?? 'license terms'}
                                </Link>
                                .
                            </Text>

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
