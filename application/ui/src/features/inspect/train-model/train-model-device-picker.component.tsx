import { useEffect } from 'react';

import { Link } from '@adobe/react-spectrum';
import { Flex, Heading, InlineAlert, Radio, RadioGroup, Text } from '@geti/ui';

import { getDeviceMetadata, selectPreferredDevice } from './utils/device-metadata';

import classes from './train-model.module.scss';

interface TrainModelDevicePickerProps {
    devices: string[];
    selectedDevice: string | null;
    onSelect: (device: string | null) => void;
}

export const TrainModelDevicePicker = ({ devices, selectedDevice, onSelect }: TrainModelDevicePickerProps) => {
    const hasDevices = devices.length > 0;

    useEffect(() => {
        if (!hasDevices) {
            if (selectedDevice !== null) {
                onSelect(null);
            }
            return;
        }

        const preferredDevice = selectPreferredDevice(devices);

        if (selectedDevice === null) {
            if (preferredDevice !== null || devices.length > 0) {
                onSelect(preferredDevice ?? devices[0]);
            }
            return;
        }

        if (!devices.includes(selectedDevice)) {
            if (preferredDevice !== null || devices.length > 0) {
                onSelect(preferredDevice ?? devices[0]);
            } else {
                onSelect(null);
            }
        }
    }, [devices, hasDevices, onSelect, selectedDevice]);

    if (!hasDevices) {
        return (
            <Flex direction='column' gap='size-150' UNSAFE_className={classes.deviceSection}>
                <Heading level={4} margin={0}>
                    Select training device
                </Heading>
                <InlineAlert variant='notice'>
                    <Heading level={5}>No training devices detected</Heading>
                    <Text>
                        Connect an available device to start a training job. If you believe this is an error,{' '}
                        <Link
                            isQuiet
                            href='https://github.com/open-edge-platform/anomalib/issues'
                            target='_blank'
                            rel='noreferrer noopener'
                        >
                            let us know on GitHub
                        </Link>
                        .
                    </Text>
                </InlineAlert>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap='size-150' UNSAFE_className={classes.deviceSection}>
            <Heading level={4} margin={0}>
                Select training device
            </Heading>
            <RadioGroup
                aria-label='Select a training device'
                orientation={devices.length > 3 ? 'vertical' : 'horizontal'}
                value={selectedDevice ?? undefined}
                onChange={(value) => {
                    onSelect(value);
                }}
                isEmphasized
                UNSAFE_className={classes.deviceGroup}
            >
                <Flex direction={devices.length > 3 ? 'column' : 'row'} gap='size-200'>
                    {devices.map((device) => {
                        const { label, description } = getDeviceMetadata(device);

                        return (
                            <Radio key={device} value={device} UNSAFE_className={classes.deviceOption}>
                                <Flex direction='column' gap='size-50'>
                                    <Text>{label}</Text>
                                    {description ? (
                                        <Text UNSAFE_className={classes.deviceDescription}>{description}</Text>
                                    ) : null}
                                </Flex>
                            </Radio>
                        );
                    })}
                </Flex>
            </RadioGroup>
        </Flex>
    );
};
