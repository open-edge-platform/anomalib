// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'failed';

export type StatusVariant = 'info' | 'success' | 'warning' | 'error';

export type StatusType = 'training' | 'export' | 'upload';

export interface MainStatusState {
    /** Unique identifier for deduplication */
    id: string;
    /** Type of operation - used for priority ordering */
    type: StatusType;
    /** Primary message */
    message: string;
    /** Secondary detail text (optional) */
    detail?: string;
    /**
     * Progress value:
     * - 0-100: determinate progress
     * - undefined: indeterminate (pulsing animation)
     */
    progress?: number;
    /** Visual style variant - controls background color */
    variant: StatusVariant;
    /** Show cancel button */
    isCancellable?: boolean;
    /** Cancel callback */
    onCancel?: () => void;
    /** Auto-remove status after delay (ms) */
    autoRemoveDelay?: number;
}

/** Priority order for main status: training > export > upload */
export const STATUS_PRIORITY: Record<StatusType, number> = {
    training: 1,
    export: 2,
    upload: 3,
};
