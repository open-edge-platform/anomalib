import { defineConfig, devices } from '@playwright/test';

const CI = !!process.env.CI;
const AUTO_START_BACKEND = process.env.AUTO_START_BACKEND === 'true';
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';

export default defineConfig({
    testDir: './tests',
    fullyParallel: false,
    forbidOnly: CI,
    retries: CI ? 2 : 0,
    workers: CI ? 1 : 1,
    reporter: [[CI ? 'github' : 'list'], ['html', { open: 'never' }]],
    timeout: 5 * 60 * 1000,
    expect: {
        timeout: 30 * 1000,
    },
    use: {
        baseURL: 'http://localhost:3000',
        trace: CI ? 'on-first-retry' : 'on',
        video: CI ? 'on-first-retry' : 'on',
        launchOptions: {
            slowMo: 100,
            headless: true,
            devtools: false,
        },
        timezoneId: 'UTC',
        actionTimeout: 30 * 1000,
        navigationTimeout: 30 * 1000,
    },

    projects: [
        {
            name: 'E2E Miscellaneous',
            testMatch: ['**/dataset-management.spec.ts', '**/project-management.spec.ts'],
            timeout: 2 * 60 * 1000,
            fullyParallel: false,
            use: { ...devices['Desktop Chrome'] },
        },
        {
            name: 'E2E Full Workflow',
            testMatch: ['**/full-workflow.spec.ts'],
            timeout: 20 * 60 * 1000,
            fullyParallel: false,
            retries: 0,
            use: { ...devices['Desktop Chrome'] },
        },
    ],

    webServer: [
        {
            command: 'npx serve -c serve.json -p 3000',
            url: 'http://localhost:3000',
            reuseExistingServer: !CI,
        },
        ...(AUTO_START_BACKEND
            ? [
                  {
                      command: './run.sh',
                      cwd: '../backend',
                      url: `${API_BASE_URL}/api/projects`,
                      reuseExistingServer: !CI,
                      timeout: 60 * 1000,
                  },
              ]
            : []),
    ],
});
