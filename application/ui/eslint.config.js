// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FlatCompat } from '@eslint/eslintrc';
import js from '@eslint/js';

import sharedEslintConfig from './eslint.shared.config.js';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const currentYear = new Date().getFullYear();
const allowedYears = Array.from({ length: Math.max(currentYear - 2025 + 1, 1) }, (_, index) => 2025 + index);
const compat = new FlatCompat({
    baseDirectory: dirname,
    recommendedConfig: js.configs.recommended,
    allConfig: js.configs.all,
});

const restrictedImportPaths = [
    {
        name: '@adobe/react-spectrum',
        message: 'Use component from the @geti-ui/ui package instead.',
    },
];

const restrictedImportPatterns = [
    {
        group: ['@react-spectrum'],
        message: 'Use component from the @geti-ui/ui package instead.',
    },
    {
        group: ['@react-types/*'],
        message: 'Use type from the @geti-ui/ui package instead.',
    },
    {
        group: ['@spectrum-icons'],
        message: 'Use icons from the @geti-ui/ui/icons package instead.',
    },
    {
        group: ['src/*'],
        message: 'Use relative imports instead of absolute "src/" imports.',
    },
];

// Only `*.tauri.{ts,tsx}` files should import `@tauri-apps/*` directly.
const tauriRestrictedImportPattern = {
    group: ['@tauri-apps/*'],
    message:
        'Import Tauri plugins only from `*.tauri.{ts,tsx}` files. Consumers should ' +
        'import the capability module so the bundler can swap implementations per build target.',
};

// Only src/api/** and src/query-client/** may import the generated spec directly.
const openapiSpecRestrictedImportPattern = {
    group: ['**/openapi-spec'],
    message: 'Do not import generated OpenAPI types directly. Import them from `@anomalib-studio/api/spec` instead.',
};

export default [
    {
        ignores: [...sharedEslintConfig[0].ignores, 'src/api/openapi-spec.d.ts'],
    },
    ...sharedEslintConfig,
    {
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: restrictedImportPaths,
                    patterns: restrictedImportPatterns,
                },
            ],
            'no-restricted-syntax': [
                'error',
                {
                    selector: "CallExpression[callee.name='isTauri']",
                    message:
                        'Do not branch on `isTauri()` at runtime. ' +
                        'Add or split a capability module via a `.tauri.{ts,tsx}` twin instead.',
                },
            ],
        },
    },
    {
        files: ['**/*.test.ts', '**/*.test.tsx', '**/*mock*.ts', '**/*.spec.ts'],
        rules: {
            'max-len': ['off'],
        },
    },
    {
        // Every source file *except* `.tauri.{ts,tsx}` twins must not import `@tauri-apps/*` directly.
        files: ['src/**/*.{ts,tsx}'],
        ignores: ['src/**/*.tauri.{ts,tsx}'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: restrictedImportPaths,
                    patterns: [...restrictedImportPatterns, tauriRestrictedImportPattern],
                },
            ],
        },
    },
    {
        files: ['src/**/*.{ts,tsx}'],
        ignores: ['src/**/*.tauri.{ts,tsx}', 'src/api/**', 'src/query-client/**'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: restrictedImportPaths,
                    patterns: [
                        ...restrictedImportPatterns,
                        tauriRestrictedImportPattern,
                        openapiSpecRestrictedImportPattern,
                    ],
                },
            ],
        },
    },
    ...compat.extends('plugin:playwright/playwright-test').map((config) => ({
        ...config,
        files: ['tests/**/*.ts'],
    })),
    {
        files: ['tests/**/*.ts'],

        rules: {
            'playwright/no-wait-for-selector': ['off'],
            'playwright/no-conditional-expect': ['off'],
            'playwright/no-standalone-expect': ['off'],
            'playwright/missing-playwright-await': ['warn'],
            'playwright/valid-expect': ['warn'],
            'playwright/no-useless-not': ['warn'],
            'playwright/no-page-pause': ['warn'],
            'playwright/prefer-to-have-length': ['warn'],
            'playwright/no-conditional-in-test': ['off'],
            'playwright/expect-expect': ['off'],
            'playwright/no-skipped-test': ['off'],
            'playwright/no-wait-for-timeout': ['off'],
            'playwright/no-nested-step': ['off'],
        },
    },
];
