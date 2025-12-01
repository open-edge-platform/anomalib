import { isString } from 'lodash-es';

export const removeUnderscore = (text: string) => {
    return text.replaceAll('_', ' ');
};

export const isStatusActive = (status: string) => {
    return ['running', 'active'].includes(status);
};

export const formatSize = (bytes: number | null | undefined) => {
    if (bytes === null || bytes === undefined) {
        return '';
    }

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }

    const maximumFractionDigits = size >= 10 ? 0 : 1;
    const formatter = new Intl.NumberFormat(undefined, { maximumFractionDigits });

    return `${formatter.format(size)} ${units[unitIndex]}`;
};

export const isNonEmptyString = (value: unknown): value is string => isString(value) && value !== '';
