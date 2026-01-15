interface DeviceMetadata {
    label: string;
    description?: string;
}

const DEVICE_DISPLAY_NAMES: Record<string, DeviceMetadata> = {
    CPU: {
        label: 'CPU',
        description: 'High compatibility. Can be slow for large models.',
    },
    XPU: {
        label: 'Intel XPU',
        description: 'Intel unified accelerator architecture.',
    },
    GPU: {
        label: 'GPU (CUDA)',
        description: 'Accelerated training on NVIDIA CUDA-capable GPUs.',
    },
    CUDA: {
        label: 'GPU (CUDA)',
        description: 'Accelerated training on NVIDIA CUDA-capable GPUs.',
    },
    TPU: {
        label: 'TPU',
        description: 'Google Cloud Tensor Processing Units via XLA.',
    },
    XLA: {
        label: 'XLA',
        description: 'XLA-backed accelerator, commonly used for TPUs.',
    },
    HPU: {
        label: 'Habana Gaudi (HPU)',
        description: 'Optimized for Intel Habana Gaudi accelerators.',
    },
    MPS: {
        label: 'MPS (Apple Silicon GPU)',
        description: 'Apple Metal Performance Shaders accelerator for macOS.',
    },
    NPU: {
        label: 'NPU',
        description: 'Neural Processing Unit for edge and embedded deployments.',
    },
};

const DEVICE_PRIORITY = ['XPU', 'GPU', 'CUDA', 'TPU', 'XLA', 'HPU', 'MPS', 'NPU', 'CPU'];

const normalizeDevice = (device: string) => device.toUpperCase();

export const getDeviceMetadata = (device: string): DeviceMetadata => {
    const normalizedKey = normalizeDevice(device);

    return DEVICE_DISPLAY_NAMES[normalizedKey] ?? { label: device };
};

export const selectPreferredDevice = (devices: string[]): string | null => {
    if (devices.length === 0) {
        return null;
    }

    const normalizedDevices = devices.map((device) => ({ original: device, normalized: normalizeDevice(device) }));

    for (const preferred of DEVICE_PRIORITY) {
        const match = normalizedDevices.find(({ normalized }) => normalized === preferred);

        if (match) {
            return match.original;
        }
    }

    return devices[0];
};
