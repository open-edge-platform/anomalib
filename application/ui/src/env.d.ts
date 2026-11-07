/// <reference types="@rsbuild/core/types" />

// We need these two to be able to import/export svgs as ReactComponent,
// for instance: export { ReactComponent as Logo } from './logo.svg';
declare module '*.svg' {
    export const ReactComponent: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
}
