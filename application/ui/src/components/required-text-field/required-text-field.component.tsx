// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, useRef } from 'react';

import { TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

interface RequiredTextFieldProps extends Omit<ComponentProps<typeof TextField>, 'onFocus' | 'isRequired' | 'validate'> {
    errorMessage: string;
}

export const RequiredTextField = ({ errorMessage, ...props }: RequiredTextFieldProps) => {
    const isTouched = useRef(false);

    return (
        <TextField
            {...props}
            isRequired
            onFocus={() => (isTouched.current = true)}
            validate={(value) => (isEmpty(value.trim()) && isTouched.current ? errorMessage : '')}
        />
    );
};
