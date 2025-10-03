import { defineConfig, loadEnv } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSass } from '@rsbuild/plugin-sass';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

const { publicVars } = loadEnv({ prefixes: ['PUBLIC_'] });

export default defineConfig({
    plugins: [
        pluginReact(),

        pluginSass(),

        pluginSvgr({
            svgrOptions: {
                exportType: 'named',
            },
        }),
    ],

    source: {
        define: {
            ...publicVars,
            'import.meta.env.PUBLIC_API_BASE_URL':
                publicVars['import.meta.env.PUBLIC_API_BASE_URL'] ?? '"http://localhost:8000"',
            'process.env.PUBLIC_API_BASE_URL':
                publicVars['process.env.PUBLIC_API_BASE_URL'] ?? '"http://localhost:8000"',
            // Needed to prevent an issue with spectrum's picker
            // eslint-disable-next-line max-len
            // https://github.com/adobe/react-spectrum/blob/6173beb4dad153aef74fc81575fd97f8afcf6cb3/packages/%40react-spectrum/overlays/src/OpenTransition.tsx#L40
            'process.env': {},
        },
    },
    html: {
        title: 'Geti Inspect',
        favicon: './src/assets/icons/build-icon.svg',
    },
    tools: {
        rspack: {
            watchOptions: {
                ignored: ['**/src-tauri/**'],
            },
        },
    },
    server: {
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
                ws: true,
                //pathRewrite: { '^/api': '' }, // strip the /api prefix
            },
        },
    },
});
