import { ErrorIcon, SuccessIcon } from '@geti-inspect/icons';
import { toast, Toaster } from 'sonner';

import classes from './notification.module.scss';

type NotificationType = 'success' | 'error' | 'info';

const DEFAULT_TIME_ON_SCREEN = 5000;

export const notify = (type: NotificationType, text: string) => {
    switch (type) {
        case 'info':
            toast.info(text, {
                unstyled: true,
                style: {
                    '--border-color': 'var(--background-inverse)',
                },
            });
            break;

        case 'success':
            toast.success(text, {
                unstyled: true,
                style: {
                    '--border-color': 'var(--moss-tint-1)',
                },
            });
            break;

        case 'error':
            toast.error(text, {
                unstyled: true,
                style: {
                    '--border-color': 'var(--coral)',
                },
            });
            break;
    }
};

export const Notification = () => {
    return (
        <Toaster
            position='bottom-center'
            duration={DEFAULT_TIME_ON_SCREEN}
            closeButton={true}
            icons={{
                error: <ErrorIcon color={'var(--coral)'} />,
                success: <SuccessIcon color={'var(--moss-tint-1)'} />,
            }}
            toastOptions={{
                classNames: {
                    toast: classes.toast,
                    closeButton: classes.closeButton,
                    icon: classes.icon,
                    content: classes.content,
                },
            }}
        />
    );
};
