// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { AlertDialog, DialogContainer, Flex, IntelBrandedLoading, Link, Text } from '@geti/ui';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { getApiUrl } from '../../api/client';

type LicenseInfo = {
    distribution_license_name: string;
    distribution_license_url: string;
    source_license_name: string;
    source_license_url: string;
    third_party_notices_url: string;
};

type LicenseStatus = {
    accepted: boolean;
    app_version: string;
    is_desktop: boolean;
    license: LicenseInfo | null;
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
                            {licenseStatus.license ? (
                                <>
                                    <Text>
                                        This application is distributed under the{' '}
                                        <Link
                                            href={licenseStatus.license.distribution_license_url}
                                            target='_blank'
                                            rel='noreferrer'
                                        >
                                            {licenseStatus.license.distribution_license_name}
                                        </Link>
                                        .
                                    </Text>
                                    <Text>
                                        The source code is licensed under the{' '}
                                        <Link
                                            href={licenseStatus.license.source_license_url}
                                            target='_blank'
                                            rel='noreferrer'
                                        >
                                            {licenseStatus.license.source_license_name}
                                        </Link>
                                        . Some components are distributed under their own license terms. See the{' '}
                                        <Link
                                            href={licenseStatus.license.third_party_notices_url}
                                            target='_blank'
                                            rel='noreferrer'
                                        >
                                            Third Party Programs
                                        </Link>{' '}
                                        file for details.
                                    </Text>
                                </>
                            ) : (
                                <Text>Please accept the license terms to continue.</Text>
                            )}

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
