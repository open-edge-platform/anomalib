/**
 * Calculates the actual rendered dimensions and position of an image element when using object-fit: contain.
 *
 * This function determines how an image is displayed within its container when the image maintains
 * its aspect ratio and is scaled to fit entirely within the container bounds. It calculates the
 * actual rendered size and the offset position where the image appears within the container.
 */

export const getImageDimensions = (img: HTMLImageElement) => {
    const containerWidth = img.clientWidth;
    const containerHeight = img.clientHeight;

    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    if (naturalHeight === 0 || containerHeight === 0) {
        return { top: 0, left: 0, width: 0, height: 0 };
    }

    const imageRatio = naturalWidth / naturalHeight;
    const containerRatio = containerWidth / containerHeight;

    let renderedWidth, renderedHeight;

    if (imageRatio > containerRatio) {
        renderedWidth = containerWidth;
        renderedHeight = containerWidth / imageRatio;
    } else {
        renderedHeight = containerHeight;
        renderedWidth = containerHeight * imageRatio;
    }

    // Calculate offset (image is centered by default with object-fit)
    const offsetTop = (containerHeight - renderedHeight) / 2;
    const offsetLeft = (containerWidth - renderedWidth) / 2;

    return {
        top: offsetTop,
        left: offsetLeft,
        width: renderedWidth,
        height: renderedHeight,
    };
};
