// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex } from '@geti/ui';
import { Play } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { isEmpty, noop } from 'lodash-es';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import classes from './play-stream-button.module.scss';

type PlayStreamButtonProps = {
    onStart?: () => void;
    isDisabled?: boolean;
};

export const PlayStreamButton = ({ isDisabled: isDisabledByProp = false, onStart }: PlayStreamButtonProps) => {
    const { data: pipeline } = usePipeline();

    const isDisabled = isDisabledByProp || isEmpty(pipeline?.source);

    return (
        <div
            className={clsx(classes.container, { [classes.disabled]: isDisabled })}
            onClick={isDisabled ? noop : onStart}
        >
            <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                <Button
                    onPress={onStart}
                    aria-label={'Start stream'}
                    isDisabled={isDisabled}
                    UNSAFE_className={classes.playButton}
                >
                    <Play width='20px' height='20px' />
                </Button>
            </Flex>
        </div>
    );
};
