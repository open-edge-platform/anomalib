export const downloadFile = (url: string, name?: string) => {
    const link = document.createElement('a');

    if (name) {
        link.download = name;
    }

    link.href = url;
    link.hidden = true;
    link.click();

    setTimeout(() => link.remove(), 100);
};
