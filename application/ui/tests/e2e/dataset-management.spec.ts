import path from 'path';

import { expect, test, TEST_ASSETS_DIR } from '../fixtures';

test.describe('Dataset Management', () => {
    test('uploads images and displays them in the dataset list', async ({ page, createProject }) => {
        const project = await createProject('Upload Test');

        await page.goto(`/projects/${project.id}?mode=Dataset`);

        await expect(page.getByRole('button', { name: /upload images/i })).toBeVisible();

        const imagePaths = [1, 2, 3, 4, 5].map((i) =>
            path.join(TEST_ASSETS_DIR, 'normal', `${String(i).padStart(3, '0')}.png`)
        );

        const fileInput = page.locator('input[type="file"]');
        await fileInput.setInputFiles(imagePaths);

        await expect(async () => {
            const images = page.locator('[aria-label="sidebar-items"] img');
            const count = await images.count();
            expect(count).toBeGreaterThanOrEqual(5);
        }).toPass({ timeout: 60 * 1000 });
    });
});
