import path from 'path';
import { fileURLToPath } from 'url';

import { test as base, expect, APIRequestContext, Page } from '@playwright/test';
import { randomUUID } from 'crypto';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';

const currentDir = path.dirname(fileURLToPath(import.meta.url));
export const TEST_ASSETS_DIR = path.join(currentDir, 'e2e', 'test-assets');

interface TestProject {
    id: string;
    name: string;
}

interface TestFixtures {
    api: APIRequestContext;
    createProject: (name?: string) => Promise<TestProject>;
    deleteProject: (projectId: string) => Promise<void>;
    clearAllProjects: () => Promise<void>;
    getProjectIdFromUrl: (page: Page) => string | null;
    cleanupProjects: string[];
}

export const test = base.extend<TestFixtures>({
    api: async ({ playwright }, use) => {
        const context = await playwright.request.newContext({
            baseURL: API_BASE_URL,
        });
        await use(context);
        await context.dispose();
    },

    cleanupProjects: async ({}, use) => {
        const projectIds: string[] = [];
        await use(projectIds);
    },

    createProject: async ({ api, cleanupProjects }, use) => {
        const createFn = async (name?: string): Promise<TestProject> => {
            const projectId = randomUUID();
            const projectName = name || `Test Project ${projectId.slice(0, 8)}`;

            const response = await api.post('/api/projects', {
                data: { id: projectId, name: projectName },
            });

            if (!response.ok()) {
                const body = await response.text();
                throw new Error(`Failed to create project: ${response.status()} - ${body}`);
            }

            cleanupProjects.push(projectId);
            return { id: projectId, name: projectName };
        };

        await use(createFn);
    },

    deleteProject: async ({ api }, use) => {
        const deleteFn = async (projectId: string): Promise<void> => {
            await api.delete(`/api/projects/${projectId}`);
        };
        await use(deleteFn);
    },

    clearAllProjects: async ({ api }, use) => {
        const clearFn = async (): Promise<void> => {
            const projectsResp = await api.get('/api/projects');
            const { projects } = await projectsResp.json();
            for (const p of projects) {
                // Cancel any running training jobs first
                const modelsResp = await api.get(`/api/projects/${p.id}/models`);
                const { models } = await modelsResp.json();
                for (const m of models) {
                    if (m.status === 'training') {
                        await api.post(`/api/projects/${p.id}/models/${m.id}/cancel`);
                    }
                }
                await api.delete(`/api/projects/${p.id}`);
            }
        };
        await use(clearFn);
    },

    getProjectIdFromUrl: async ({}, use) => {
        const getFn = (page: Page): string | null => {
            const url = page.url();
            const match = url.match(/\/projects\/([^/?]+)/);
            return match?.[1] ?? null;
        };
        await use(getFn);
    },
});

test.afterEach(async ({ api, cleanupProjects }) => {
    for (const projectId of cleanupProjects) {
        try {
            await api.delete(`/api/projects/${projectId}`);
        } catch {
            // Ignore cleanup errors
        }
    }
});

export { expect };
