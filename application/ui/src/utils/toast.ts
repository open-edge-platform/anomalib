import { toast as toastApi, type ToastOptions } from '@geti-ui/ui';

interface LegacyToastInput {
    type?: 'success' | 'error' | 'info' | 'neutral';
    message: string;
    id?: string;
    title?: string;
    duration?: number;
    timeout?: number;
    actionButtons?: unknown[];
    position?: string;
}

export const toast = ({ type = 'neutral', message, duration, timeout, ...rest }: LegacyToastInput) => {
    const options: ToastOptions = { ...rest, timeout: timeout ?? duration };
    switch (type) {
        case 'success':
            return toastApi.positive(message, options);
        case 'error':
            return toastApi.negative(message, options);
        case 'info':
            return toastApi.info(message, options);
        default:
            return toastApi.neutral(message, options);
    }
};
