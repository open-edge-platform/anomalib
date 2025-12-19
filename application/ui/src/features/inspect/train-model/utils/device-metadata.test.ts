// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { selectPreferredDevice } from './device-metadata';

describe('selectPreferredDevice', () => {
    it('returns null for empty device list', () => {
        expect(selectPreferredDevice([])).toBeNull();
    });

    it('prefers XPU over CPU', () => {
        expect(selectPreferredDevice(['CPU', 'XPU'])).toBe('XPU');
        expect(selectPreferredDevice(['xpu', 'cpu'])).toBe('xpu');
    });

    it('prefers GPU/CUDA over CPU', () => {
        expect(selectPreferredDevice(['CPU', 'GPU'])).toBe('GPU');
        expect(selectPreferredDevice(['CPU', 'CUDA'])).toBe('CUDA');
        expect(selectPreferredDevice(['cuda', 'cpu'])).toBe('cuda');
    });

    it('returns first device if no preferred devices found', () => {
        expect(selectPreferredDevice(['unknown1', 'unknown2'])).toBe('unknown1');
        expect(selectPreferredDevice(['custom-device'])).toBe('custom-device');
    });

    it('respects priority order', () => {
        // XPU has highest priority
        expect(selectPreferredDevice(['CPU', 'GPU', 'XPU'])).toBe('XPU');
        // GPU before TPU
        expect(selectPreferredDevice(['CPU', 'TPU', 'GPU'])).toBe('GPU');
        // MPS before CPU
        expect(selectPreferredDevice(['CPU', 'MPS'])).toBe('MPS');
    });
});
