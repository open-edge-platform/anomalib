import { expect, test } from '../fixtures';

test.describe('Project Management', () => {
    // Run tests serially to avoid race conditions
    test.describe.configure({ mode: 'serial' });

    test('creates additional project via add button', async ({ page, createProject, clearAllProjects }) => {
        await clearAllProjects();

        const project = await createProject('First Project');

        await page.goto(`/projects/${project.id}`);
        await page.waitForLoadState('domcontentloaded');

        await page.getByRole('button', { name: /Selected project First Project/i }).click();
        await expect(page.getByRole('dialog')).toBeVisible();

        const list = page.getByRole('list');
        const initialItems = await list.locator('li').count();
        expect(initialItems).toBe(1);

        await page.getByRole('button', { name: /add project/i }).click();

        await expect(list.locator('li')).toHaveCount(2);

        await expect(page.getByRole('textbox', { name: /edit project name/i })).toBeVisible();
    });

    test('renames project via actions menu', async ({ page, createProject }) => {
        const project = await createProject('Original Name');

        await page.goto(`/projects/${project.id}`);
        await page.waitForLoadState('domcontentloaded');

        await page.getByRole('button', { name: /Selected project Original Name/i }).click();
        await expect(page.getByRole('dialog')).toBeVisible();

        const projectItem = page.getByRole('list').locator('li').filter({ hasText: 'Original Name' });
        await projectItem.getByRole('button', { name: /project actions/i }).click();
        await page.getByRole('menuitem', { name: /rename/i }).click();

        const input = page.getByRole('textbox', { name: /edit project name/i });
        await input.clear();
        await input.fill('Renamed Project');
        await input.press('Enter');

        await expect(page.getByRole('list').getByText('Renamed Project')).toBeVisible();
    });

    test('deletes project via actions menu', async ({ page, createProject }) => {
        const project1 = await createProject('Project To Keep');
        await createProject('Project To Remove');

        await page.goto(`/projects/${project1.id}`);
        await page.waitForLoadState('domcontentloaded');

        await page.getByRole('button', { name: /Selected project Project To Keep/i }).click();
        await expect(page.getByRole('dialog')).toBeVisible();

        const list = page.getByRole('list');
        await expect(list.getByText('Project To Keep')).toBeVisible();
        await expect(list.getByText('Project To Remove')).toBeVisible();

        const projectToDeleteItem = list.locator('li').filter({ hasText: 'Project To Remove' });
        await projectToDeleteItem.getByRole('button', { name: /project actions/i }).click();

        await page.getByRole('menuitem', { name: /delete/i }).click();

        await expect(page.getByRole('alertdialog')).toBeVisible();
        await page.getByRole('button', { name: /^Delete$/i }).click();

        await expect(list.getByText('Project To Remove')).toBeHidden();
        await expect(list.getByText('Project To Keep')).toBeVisible();
    });

    test('switches between projects', async ({ page, createProject }) => {
        const project1 = await createProject('Project Alpha');
        const project2 = await createProject('Project Beta');

        await page.goto(`/projects/${project1.id}`);
        await page.waitForLoadState('domcontentloaded');
        await expect(page).toHaveURL(new RegExp(project1.id));

        await page.getByRole('button', { name: /Selected project Project Alpha/i }).click();
        await expect(page.getByRole('dialog')).toBeVisible();

        await page.getByRole('list').locator('li').filter({ hasText: 'Project Beta' }).click();
        await expect(page).toHaveURL(new RegExp(project2.id));

        await page.keyboard.press('Escape');
        await expect(page.getByRole('dialog')).toBeHidden({ timeout: 5000 });
        await page.waitForLoadState('domcontentloaded');

        await page.getByRole('button', { name: /Selected project Project Beta/i }).click();
        await expect(page.getByRole('dialog')).toBeVisible();
        await page.getByRole('list').locator('li').filter({ hasText: 'Project Alpha' }).click();
        await expect(page).toHaveURL(new RegExp(project1.id));
    });
});

// Separate describe block for tests that require empty database state
// These are more fragile due to needing no projects to exist
test.describe('Project Management - Welcome Page', () => {
    test.describe.configure({ mode: 'serial' });

    test('creates project from welcome page when no projects exist', async ({ page, clearAllProjects }) => {
        // Delete all projects - this test MUST run in isolation
        await clearAllProjects();
        // Wait for deletions to process
        await page.waitForTimeout(1000);

        // Navigate fresh
        await page.goto('/');

        // Should redirect to welcome page when no projects exist
        await expect(page).toHaveURL(/\/welcome/, { timeout: 15000 });
        await expect(page.getByRole('heading', { name: /Welcome to Geti Inspect/i })).toBeVisible();

        await page.getByRole('button', { name: /create project/i }).click();

        await expect(page).toHaveURL(/\/projects\/.+/);
        // The project should be created with name "Project #1" - check the header button
        await expect(page.getByRole('button', { name: /Selected project Project #1/i })).toBeVisible();
    });

    test('navigates to dataset mode after project creation', async ({ page, clearAllProjects }) => {
        await clearAllProjects();
        await page.waitForTimeout(1000);

        await page.goto('/');
        // Should redirect to welcome page when no projects exist
        await expect(page).toHaveURL(/\/welcome/, { timeout: 15000 });
        await page.getByRole('button', { name: /create project/i }).click();

        await expect(page).toHaveURL(/mode=Dataset/);
    });
});
