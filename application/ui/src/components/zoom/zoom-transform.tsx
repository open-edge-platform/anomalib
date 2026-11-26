import { ReactNode, useEffect, useMemo, useRef } from 'react';

import { useContainerSize } from './use-container-size';
import { useSetZoom, useZoom } from './zoom';

import classes from './zoom.module.scss';

type Size = { width: number; height: number };

const DEFAULT_SCREEN_ZOOM = 0.9;
const getCenterCoordinates = (container: Size, target: Size) => {
    // Scale image so that it fits perfectly in the container
    const scale = DEFAULT_SCREEN_ZOOM * Math.min(container.width / target.width, container.height / target.height);

    return {
        scale,
        // Center image
        translate: {
            x: container.width / 2 - target.width / 2,
            y: container.height / 2 - target.height / 2,
        },
    };
};

const INITIAL_ZOOM = { scale: 1.0, translate: { x: 0, y: 0 } };
const SyncZoom = ({ container, target }: { container: Size; target: Size }) => {
    const setZoom = useSetZoom();

    const targetZoom = useMemo(() => {
        if (container.width === undefined || container.height === undefined) {
            return INITIAL_ZOOM;
        }

        return getCenterCoordinates({ width: container.width, height: container.height }, target);
    }, [container, target]);

    useEffect(() => {
        setZoom({
            scale: Number(targetZoom.scale.toFixed(3)),
            translate: {
                x: Number(targetZoom.translate.x.toFixed(3)),
                y: Number(targetZoom.translate.y.toFixed(3)),
            },
        });
    }, [targetZoom.scale, targetZoom.translate.x, targetZoom.translate.y, setZoom]);

    return null;
};

export const ZoomTransform = ({ children, target }: { children: ReactNode; target: Size }) => {
    const zoom = useZoom();
    const ref = useRef<HTMLDivElement>(null);
    const containerSize = useContainerSize(ref);

    return (
        <div
            ref={ref}
            className={classes.wrapper}
            style={{
                // Enable hardware acceleration
                transform: 'translate3d(0, 0, 0)',
                '--zoom-scale': zoom.scale,
            }}
        >
            <div
                data-testid='zoom-transform'
                className={classes.wrapperInternal}
                style={{
                    transform: `translate(${zoom.translate.x}px, ${zoom.translate.y}px) scale(${zoom.scale})`,
                }}
            >
                <SyncZoom container={containerSize} target={target} />
                {children}
            </div>
        </div>
    );
};
