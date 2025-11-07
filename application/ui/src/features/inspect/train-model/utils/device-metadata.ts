export interface DeviceMetadata {
    label: string;
    description?: string;
}

const DEVICE_DISPLAY_NAMES: Record<string, DeviceMetadata> = {
    CPU: {
        label: 'CPU',
        description: 'High compatibility. Can be slow for large models.',
    },
    GPU: {
        label: 'GPU (CUDA)',
        description: 'Accelerated training on NVIDIA CUDA-capable GPUs.',
    },
    MPS: {
        label: 'MPS (Apple Silicon GPU)',
        description: 'Apple Metal Performance Shaders accelerator for macOS.',
    },
    TPU: {
        label: 'TPU',
        description: 'Google Cloud Tensor Processing Units via XLA.',
    },
    XLA: {
        label: 'TPU (XLA)',
        description: 'XLA-backed accelerator, commonly used for TPUs.',
    },
    HPU: {
        label: 'Habana Gaudi (HPU)',
        description: 'Optimized for Intel Habana Gaudi accelerators.',
    },
    XPU: {
        label: 'Intel XPU',
        description: 'Intel unified accelerator architecture.',
    },
    NPU: {
        label: 'NPU',
        description: 'Neural Processing Unit for edge and embedded deployments.',
    },
};

export const getDeviceMetadata = (device: string): DeviceMetadata => {
    const normalizedKey = device.toUpperCase();

    return DEVICE_DISPLAY_NAMES[normalizedKey] ?? { label: device };
};
