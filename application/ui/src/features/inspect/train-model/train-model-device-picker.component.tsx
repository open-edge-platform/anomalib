import { useEffect, useMemo } from 'react';

import { $api } from '@geti-inspect/api';
import { Button, Content, Flex, Heading, InlineAlert, Loading, Radio, RadioGroup, Text } from '@geti/ui';

import { getDeviceMetadata } from './utils/device-metadata';

import classes from './train-model.module.scss';

interface TrainModelDevicePickerProps {
    selectedDevice: string | null;
    onSelect: (device: string | null) => void;
}

export const TrainModelDevicePicker = ({ selectedDevice, onSelect }: TrainModelDevicePickerProps) => {
    const {
        data: availableDevices,
        isLoading,
        isError,
        isRefetching,
        refetch,
    } = $api.useQuery('get', '/api/training-devices');
    const devices = useMemo(() => availableDevices?.devices ?? [], [availableDevices]);
    const hasDevices = devices.length > 0;
    const isLoadingDevices = isLoading || isRefetching;

    useEffect(() => {
        if (isLoadingDevices || isError || !hasDevices) {
            if (selectedDevice !== null) {
                onSelect(null);
            }
            return;
        }

        if (selectedDevice === null) {
            onSelect(devices[0]);
        } else if (!devices.includes(selectedDevice)) {
            onSelect(null);
        }
    }, [devices, hasDevices, isError, isLoadingDevices, onSelect, selectedDevice]);

    const deviceOptions = useMemo(() => {
        return devices.map((device) => {
            const { label, description } = getDeviceMetadata(device);

            return {
                value: device,
                label,
                description,
            };
        });
    }, [devices]);

    return (
        <Flex direction='column' gap='size-150' UNSAFE_className={classes.deviceSection}>
            <Heading level={4} margin={0}>
                Select training device
            </Heading>
            {isLoadingDevices ? (
                <Flex alignItems='center' justifyContent='center' gap='size-150'>
                    <Loading mode='inline' size='M' />
                    <Text>Loading available devices…</Text>
                </Flex>
            ) : null}
            {isError ? (
                <InlineAlert variant='negative'>
                    <Heading level={5}>Unable to load training devices</Heading>
                    <Content>
                        <Flex alignItems='center' gap='size-150'>
                            <Text>Check your connection or retry.</Text>
                            <Button
                                variant='secondary'
                                onPress={() => {
                                    void refetch();
                                }}
                            >
                                Retry
                            </Button>
                        </Flex>
                    </Content>
                </InlineAlert>
            ) : null}
            {!isLoadingDevices && !isError && !hasDevices ? (
                <InlineAlert variant='notice'>
                    <Heading level={5}>No training devices detected</Heading>
                    <Content>Connect an available device to start a training job.</Content>
                </InlineAlert>
            ) : null}
            {!isLoadingDevices && !isError && hasDevices ? (
                <RadioGroup
                    aria-label='Select a training device'
                    orientation={deviceOptions.length > 3 ? 'vertical' : 'horizontal'}
                    value={selectedDevice ?? undefined}
                    onChange={(value) => {
                        onSelect(value);
                    }}
                    isEmphasized
                    UNSAFE_className={classes.deviceGroup}
                >
                    <Flex direction={deviceOptions.length > 3 ? 'column' : 'row'} gap='size-200'>
                        {deviceOptions.map((device) => (
                            <Radio key={device.value} value={device.value} UNSAFE_className={classes.deviceOption}>
                                <Flex direction='column' gap='size-50'>
                                    <Text>{device.label}</Text>
                                    {device.description ? (
                                        <Text UNSAFE_className={classes.deviceDescription}>{device.description}</Text>
                                    ) : null}
                                </Flex>
                            </Radio>
                        ))}
                    </Flex>
                </RadioGroup>
            ) : null}
        </Flex>
    );
};
