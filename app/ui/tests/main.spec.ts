import { expect, test } from './fixtures';

test.describe('Geti Inspect', () => {
    test('Allows users to inspect', async ({ page }) => {
        await page.goto('/');
        await expect(page.getByText('Geti Inspect')).toBeVisible();
    });
});
