export const removeUnderscore = (text: string) => {
    return text.replaceAll('_', ' ');
};

export const isStatusActive = (status: string) => {
    return ['running', 'active'].includes(status);
};
