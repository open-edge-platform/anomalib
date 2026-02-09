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
                publicVars['import.meta.env.PUBLIC_API_BASE_URL'] ?? '""',
            'process.env.PUBLIC_API_BASE_URL':
                publicVars['process.env.PUBLIC_API_BASE_URL'] ?? '""',
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
            },
        },
    },
});
