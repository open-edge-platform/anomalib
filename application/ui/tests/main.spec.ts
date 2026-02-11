import { expect, test } from '@anomalib-studio/test-fixtures';

test.describe('Anomalib Studio', () => {
    test('Allows users to inspect', async ({ page }) => {
        await page.goto('/', { waitUntil: 'domcontentloaded' });

        await expect(page.getByText(/Anomalib Studio/i)).toBeVisible();
    });
});
