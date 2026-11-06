import { expect, test } from '@anomalib-studio/test-fixtures';

test.describe('Anomalib Studio', () => {
    test('Allows users to inspect', async ({ page }) => {
        await page.goto('/', { waitUntil: 'domcontentloaded' });

        await expect(page.getByText(/Anomalib Studio/i)).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Normal images', exact: true })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Train model', exact: true })).toBeDisabled();
    });
});
