import type { SchemaDeviceInfo } from 'src/api/openapi-spec';

const DEVICE_DESCRIPTIONS: Record<string, string> = {
    CPU: 'High compatibility. Can be slow for large models.',
    XPU: 'Intel unified accelerator architecture.',
    GPU: 'Accelerated training on NVIDIA CUDA-capable GPUs.',
    CUDA: 'Accelerated training on NVIDIA CUDA-capable GPUs.',
    TPU: 'Google Cloud Tensor Processing Units via XLA.',
    XLA: 'XLA-backed accelerator, commonly used for TPUs.',
    HPU: 'Optimized for Intel Habana Gaudi accelerators.',
    MPS: 'Apple Metal Performance Shaders accelerator for macOS.',
    NPU: 'Neural Processing Unit for edge and embedded deployments.',
};

const DEVICE_PRIORITY = ['XPU', 'GPU', 'CUDA', 'TPU', 'XLA', 'HPU', 'MPS', 'NPU', 'CPU'];

/**
 * Build a unique key for a device.
 * CPU/MPS/NPU (index === null) → "cpu", "mps"
 * Indexed devices → "xpu-0", "cuda-1"
 */
export const getDeviceKey = (device: SchemaDeviceInfo): string => {
    if (device.index == null) {
        return device.type;
    }
    return `${device.type}-${device.index}`;
};

/**
 * Build a human-readable label for a device.
 * Uses the device name from the API, appending [index] for indexed devices.
 * e.g. "CPU", "Intel(R) Graphics [0]", "NVIDIA RTX 4090 [1]"
 */
export const getDeviceLabel = (device: SchemaDeviceInfo): string => {
    if (device.index == null) {
        return device.name;
    }
    return `${device.name} [${device.index}]`;
};

export const getDeviceDescription = (type: string): string | undefined => {
    return DEVICE_DESCRIPTIONS[type.toUpperCase()];
};

export const selectPreferredDevice = (devices: SchemaDeviceInfo[]): string | null => {
    if (devices.length === 0) {
        return null;
    }

    for (const preferred of DEVICE_PRIORITY) {
        const match = devices.find((device) => device.type.toUpperCase() === preferred);

        if (match) {
            return getDeviceKey(match);
        }
    }

    return getDeviceKey(devices[0]);
};
