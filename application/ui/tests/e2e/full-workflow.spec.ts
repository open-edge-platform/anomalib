import fs from 'fs';
import path from 'path';

import { expect, test, TEST_ASSETS_DIR } from '../fixtures';

const TEST_VIDEO_PATH = path.join(TEST_ASSETS_DIR, 'test-video.mp4');
const TEST_OUTPUT_DIR = path.join(TEST_ASSETS_DIR, 'output');
const TEST_EXPORT_DIR = path.join(TEST_ASSETS_DIR, 'exports');

const cleanupOutputDirectory = () => {
    if (fs.existsSync(TEST_OUTPUT_DIR)) {
        const files = fs.readdirSync(TEST_OUTPUT_DIR);
        for (const file of files) {
            if (
                file.endsWith('-pred.txt') ||
                file.endsWith('-pred.json') ||
                file.endsWith('-pred.jpg') ||
                file.endsWith('-original.jpg')
            ) {
                fs.unlinkSync(path.join(TEST_OUTPUT_DIR, file));
            }
        }
    }
};

const cleanupExportDirectory = () => {
    if (fs.existsSync(TEST_EXPORT_DIR)) {
        const files = fs.readdirSync(TEST_EXPORT_DIR);
        for (const file of files) {
            if (file.endsWith('.zip')) {
                fs.unlinkSync(path.join(TEST_EXPORT_DIR, file));
            }
        }
    }
};

const ensureExportDirectoryExists = () => {
    if (!fs.existsSync(TEST_EXPORT_DIR)) {
        fs.mkdirSync(TEST_EXPORT_DIR, { recursive: true });
    }
};

test.describe('Full Workflow', () => {
    test.afterAll(() => {
        cleanupOutputDirectory();
        cleanupExportDirectory();
    });
    test.describe.configure({ mode: 'serial' });

    test('complete workflow: project creation, training, export, and pipeline configuration', async ({
        page,
        api,
        clearAllProjects,
        getProjectIdFromUrl,
    }) => {
        await clearAllProjects();
        ensureExportDirectoryExists();

        await test.step('create project from welcome page', async () => {
            await page.goto('/welcome');
            await page.waitForLoadState('domcontentloaded');
            await expect(page.getByText(/Welcome to Geti Inspect/i)).toBeVisible({ timeout: 10000 });

            await page.getByRole('button', { name: /create project/i }).click();

            await expect(page).toHaveURL(/\/projects\/.+\?mode=Dataset/);
        });

        await test.step('upload training images', async () => {
            const normalImages = Array.from({ length: 20 }, (_, i) =>
                path.join(TEST_ASSETS_DIR, 'normal', `${String(i + 1).padStart(3, '0')}.png`)
            );

            const fileInput = page.locator('input[type="file"]');
            await fileInput.setInputFiles(normalImages);

            // Wait for Train model button to become enabled (images finished uploading)
            const datasetHeader = page.getByRole('heading', { name: 'Dataset' });
            await expect(datasetHeader.getByRole('button', { name: /train model/i })).toBeEnabled({
                timeout: 2 * 60 * 1000,
            });
        });

        await test.step('open training dialog and start training', async () => {
            const datasetHeader = page.getByRole('heading', { name: 'Dataset' });
            await datasetHeader.getByRole('button', { name: /train model/i }).click();

            await expect(page.getByRole('dialog', { name: /Train model/ })).toBeVisible();

            await page.getByRole('radio', { name: /patchcore/i }).click();

            await expect(page.getByRole('button', { name: /start/i })).toBeEnabled();

            await page.getByRole('button', { name: /start/i }).click();

            await expect(page.getByRole('dialog', { name: /Train model/ })).toBeHidden({ timeout: 60 * 1000 });
        });

        await test.step('wait for training to complete', async () => {
            await page.goto(page.url().replace('mode=Dataset', 'mode=Models'));

            // Wait for training to complete - model shows "Active" status when done
            await expect(page.getByText(/Active/)).toBeVisible({
                timeout: 15 * 60 * 1000,
            });
        });

        await test.step('check training logs dialog can be opened', async () => {
            await page.getByRole('button', { name: /model actions/i }).click();
            await page.getByRole('menuitem', { name: /View logs/i }).click();

            const logsDialog = page.getByRole('dialog', { name: 'Logs' });
            await expect(logsDialog).toBeVisible();

            // Wait for log entries to load (or "No logs available" message)
            await expect(
                logsDialog.locator('[class*="logEntry"]').first().or(logsDialog.getByText(/No logs available/i))
            ).toBeVisible({ timeout: 10000 });

            await logsDialog.getByRole('button', { name: /close/i }).click();
            await expect(logsDialog).toBeHidden();
        });

        await test.step('export model and verify file is downloaded', async () => {
            await page.getByRole('link', { name: /patchcore/i }).click();

            await expect(page.getByText(/Export Model/i)).toBeVisible();
            await expect(page.getByRole('radiogroup', { name: /Select export format/i })).toBeVisible();

            // Select OpenVINO format (it's selected by default, but we explicitly click it for clarity)
            await page.getByRole('radio', { name: 'OpenVINO' }).click();

            const downloadPromise = page.waitForEvent('download');
            await page.getByRole('button', { name: /^Export$/i }).click();
            const download = await downloadPromise;

            const filename = download.suggestedFilename();
            expect(filename).toMatch(/.*_patchcore_openvino.*\.zip$/i);

            // Save the file to verify it
            const downloadPath = path.join(TEST_EXPORT_DIR, filename);
            await download.saveAs(downloadPath);

            expect(fs.existsSync(downloadPath)).toBe(true);
            const stats = fs.statSync(downloadPath);
            expect(stats.size).toBeGreaterThan(1000);

            await page.getByRole('button', { name: /back to models/i }).click();
        });

        await test.step('configure pipeline with folder output', async () => {
            await page.getByRole('button', { name: /pipeline configuration/i }).click();

            await expect(page.getByRole('tablist', { name: /Dataset import tabs/i })).toBeVisible();

            await page.getByRole('tab', { name: /Output/i }).click();

            await page.getByRole('button', { name: /Add new sink/i }).click();

            const folderHeading = page.getByRole('heading', { name: 'Folder' });
            await expect(folderHeading).toBeVisible();
            await folderHeading.getByRole('button').click();

            const folderGroup = page.getByRole('group', { name: 'Folder' });
            await folderGroup.getByRole('textbox', { name: /^Name$/i }).fill('Test Output');
            await folderGroup.getByRole('textbox', { name: /Folder Path/i }).fill(TEST_OUTPUT_DIR);

            await folderGroup.getByRole('checkbox', { name: 'Predictions', exact: true }).check();

            const addConnectBtn = folderGroup.getByRole('button', { name: /Add & Connect/i });
            await expect(addConnectBtn).toBeEnabled({ timeout: 30000 });
            await addConnectBtn.scrollIntoViewIfNeeded();

            // eslint-disable-next-line playwright/no-force-option -- Form submit blocked by toast overlay
            await addConnectBtn.click({ force: true });

            const projectId = getProjectIdFromUrl(page);

            // Wait for sink to be created via API polling instead of fixed timeout
            if (projectId) {
                await expect(async () => {
                    const sinksResp = await api.get(`/api/projects/${projectId}/sinks`);
                    const { sinks } = await sinksResp.json();
                    expect(sinks.length).toBeGreaterThan(0);
                    expect(sinks.some((s: { name: string }) => s.name === 'Test Output')).toBeTruthy();
                }).toPass({ timeout: 30 * 1000 });
            }
        });

        await test.step('configure pipeline with video source', async () => {
            const tablist = page.getByRole('tablist', { name: /Dataset import tabs/i });
            if (!(await tablist.isVisible())) {
                await page.getByRole('button', { name: /pipeline configuration/i }).click();
                await expect(tablist).toBeVisible();
            }

            await page.getByRole('tab', { name: /Input/i }).click();

            await page.getByRole('button', { name: /Add new source/i }).click();

            await page.getByRole('button', { name: /Video file/i }).click();

            await page.getByRole('textbox', { name: /^Name$/i }).fill('Test Video');

            // Upload video file through file picker
            const fileChooserPromise = page.waitForEvent('filechooser');
            await page.getByRole('button', { name: /Upload video file/i }).click();
            const fileChooser = await fileChooserPromise;
            await fileChooser.setFiles(TEST_VIDEO_PATH);

            // Wait for video upload to complete (button shows spinner during upload)
            await expect(page.getByRole('button', { name: /Upload video file/i })).toBeEnabled({ timeout: 60000 });

            // Verify video was selected in the picker (shows the filename after upload)
            const videoPicker = page.getByRole('button', { name: /Video list/i });
            await expect(videoPicker).not.toContainText(/No videos uploaded yet/);

            await page.getByRole('button', { name: /Add & Connect/i }).click();

            await expect(page.getByText('Test Video')).toBeVisible();

            await page.keyboard.press('Escape');
        });

        await test.step('configure pipeline model and inference device', async () => {
            const tablist = page.getByRole('tablist', { name: /Dataset import tabs/i });
            if (!(await tablist.isVisible())) {
                await page.getByRole('button', { name: /pipeline configuration/i }).click();
                await expect(tablist).toBeVisible();
            }

            await page.getByRole('tab', { name: /Model/i }).click();

            const modelButton = page
                .getByRole('tabpanel', { name: 'Model' })
                .getByRole('button', { name: /patchcore/i });

            await modelButton.click();

            // Wait for model selection to complete (button becomes enabled again after mutation)
            await expect(modelButton).toBeEnabled({ timeout: 10000 });

            // eslint-disable-next-line playwright/no-force-option -- Dialog overlay blocks normal clicks
            await page.getByRole('button', { name: /pipeline configuration/i }).click({ force: true });

            await expect(tablist).toBeHidden({ timeout: 5000 });

            await page.getByRole('button', { name: /inference devices/i }).click();

            const deviceOption = page.getByRole('option').first();
            await deviceOption.click();
        });

        await test.step('verify pipeline is running via API', async () => {
            const projectId = getProjectIdFromUrl(page);

            if (projectId) {
                await expect(async () => {
                    const pipelineResp = await api.get(`/api/projects/${projectId}/pipeline`);
                    const pipeline = await pipelineResp.json();
                    expect(pipeline.status).toBe('running');
                    expect(pipeline.source).toBeTruthy();
                    expect(pipeline.sink).toBeTruthy();
                    expect(pipeline.model).toBeTruthy();
                }).toPass({ timeout: 30 * 1000 });
            }
        });

        await test.step('verify WebRTC stream connection (optional)', async () => {
            // WebRTC may not work reliably in headless browser environments due to ICE negotiation
            // This step is optional - the pipeline is already verified to be running via API

            try {
                // Start stream if in idle state
                const startStreamButton = page.getByRole('button', { name: /Start stream/i });
                if (await startStreamButton.isVisible({ timeout: 2000 }).catch(() => false)) {
                    await startStreamButton.click();
                }

                // Wait for video element to be visible, with retry logic for reconnection
                const videoElement = page.locator('video[aria-label="stream player"]');

                // Try to connect with a shorter timeout since we've already verified pipeline via API
                await expect(async () => {
                    // Check if reconnect button is visible (stream failed to connect)
                    const reconnectButton = page.getByRole('button', { name: /Reconnect stream/i });
                    if (await reconnectButton.isVisible({ timeout: 1000 }).catch(() => false)) {
                        await reconnectButton.click();
                        // Wait for connection state to change (either video appears or reconnect button reappears)
                        await expect(
                            videoElement.or(page.getByText(/Connecting/i)).or(reconnectButton)
                        ).toBeVisible({ timeout: 3000 });
                    }

                    // Check that video element is now visible
                    await expect(videoElement).toBeVisible({ timeout: 3000 });
                }).toPass({ timeout: 20 * 1000, intervals: [2000, 5000] });

                // Wait for video to have actual content (not just placeholder)
                await expect(async () => {
                    const hasVideoContent = await videoElement.evaluate((video: HTMLVideoElement) => {
                        return video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
                    });
                    expect(hasVideoContent).toBe(true);
                }).toPass({ timeout: 15 * 1000 });

                // Verify stream status shows connected
                await expect(page.getByText(/Connected|Running/i).first()).toBeVisible({ timeout: 5 * 1000 });
            } catch {
                // WebRTC connection failed - this is acceptable in headless test environments
                // The pipeline is already verified to be running via API in the previous step
                console.warn(
                    'WebRTC stream connection failed - this is expected in some headless browser environments'
                );
            }
        });

        await test.step('toggle anomaly map and verify overlay is visible', async () => {
            const anomalyMapSwitch = page.getByRole('switch', { name: /anomaly map/i });
            await expect(anomalyMapSwitch).toBeVisible();
            await expect(anomalyMapSwitch).toBeEnabled();

            const isChecked = await anomalyMapSwitch.isChecked();
            if (!isChecked) {
                await anomalyMapSwitch.click();
                await expect(anomalyMapSwitch).toBeChecked();
            }

            // Verify the overlay setting was applied via API
            const projectId = getProjectIdFromUrl(page);
            if (projectId) {
                await expect(async () => {
                    const pipelineResp = await api.get(`/api/projects/${projectId}/pipeline`);
                    const pipeline = await pipelineResp.json();
                    expect(pipeline.overlay).toBe(true);
                }).toPass({ timeout: 10 * 1000 });
            }

            await anomalyMapSwitch.click();
            await expect(anomalyMapSwitch).not.toBeChecked();

            if (projectId) {
                await expect(async () => {
                    const pipelineResp = await api.get(`/api/projects/${projectId}/pipeline`);
                    const pipeline = await pipelineResp.json();
                    expect(pipeline.overlay).toBe(false);
                }).toPass({ timeout: 10 * 1000 });
            }
        });

        await test.step('wait for pipeline to generate output files', async () => {
            // Verify that any output files were created in the output directory
            // Files can be: -pred.txt (predictions), -pred.jpg (visualization), -original.jpg (original)
            // The toPass block handles retrying until files appear
            await expect(() => {
                const files = fs.readdirSync(TEST_OUTPUT_DIR);
                const outputFiles = files.filter(
                    (f) =>
                        f.endsWith('-pred.txt') ||
                        f.endsWith('-pred.jpg') ||
                        f.endsWith('-original.jpg') ||
                        f.endsWith('-pred.json')
                );
                expect(outputFiles.length).toBeGreaterThan(0);
            }).toPass({ timeout: 60 * 1000 });

            // Verify at least one file has content
            const files = fs.readdirSync(TEST_OUTPUT_DIR);
            const outputFiles = files.filter(
                (f) =>
                    f.endsWith('-pred.txt') ||
                    f.endsWith('-pred.jpg') ||
                    f.endsWith('-original.jpg') ||
                    f.endsWith('-pred.json')
            );
            if (outputFiles.length > 0) {
                const firstFile = path.join(TEST_OUTPUT_DIR, outputFiles[0]);
                const stats = fs.statSync(firstFile);
                expect(stats.size).toBeGreaterThan(0);
            }
        });

        await test.step('verify pipeline can be stopped', async () => {
            const projectId = getProjectIdFromUrl(page);

            if (projectId) {
                await api.post(`/api/projects/${projectId}/pipeline:stop`);

                await expect(async () => {
                    const pipelineResp = await api.get(`/api/projects/${projectId}/pipeline`);
                    const pipeline = await pipelineResp.json();
                    expect(pipeline.status).not.toBe('running');
                }).toPass({ timeout: 10 * 1000 });

                await expect(page.getByText(/Idle|Disconnected|Stopped/i).first()).toBeVisible({ timeout: 10 * 1000 });
            }
        });
    });
});
