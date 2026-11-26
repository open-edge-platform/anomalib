// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isOnlyDigits } from './util';

describe('isOnlyDigits', () => {
    it('returns true for numeric strings', () => {
        expect(isOnlyDigits('123')).toBe(true);
        expect(isOnlyDigits('0')).toBe(true);
        expect(isOnlyDigits('000123')).toBe(true);
    });

    it('returns false for non-numeric strings', () => {
        expect(isOnlyDigits('123a')).toBe(false);
        expect(isOnlyDigits('a123')).toBe(false);
        expect(isOnlyDigits('12 3')).toBe(false);
        expect(isOnlyDigits('')).toBe(false);
        expect(isOnlyDigits(' ')).toBe(false);
        expect(isOnlyDigits('12.3')).toBe(false);
        expect(isOnlyDigits('-123')).toBe(false);
    });
});
