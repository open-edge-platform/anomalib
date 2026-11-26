import { SchemaPipelineMetrics } from './../src/api/openapi-spec.d';

export const getMockedMetrics = (partial: Partial<SchemaPipelineMetrics>): SchemaPipelineMetrics => {
    return {
        time_window: { start: '2025-11-28T15:32:51.030059Z', end: '2025-11-28T15:33:51.030059Z', time_window: 60 },
        inference: {
            latency: {
                avg_ms: 76.51289340585073,
                min_ms: 59.030749995145015,
                max_ms: 260.70933300070465,
                p95_ms: 92.98036629625129,
                latest_ms: 71.58341699687298,
            },
            throughput: { avg_requests_per_second: 0.45, total_requests: 27, max_requests_per_second: 8.0 },
        },
        ...partial,
    };
};
