import pg from "pg";

const { Pool } = pg;

export const pool = new Pool({
  connectionString:
    process.env.DATABASE_URL ||
    "postgres://santiye:santiye123@localhost:5432/santiye_guvenlik"
});

export async function initDb() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS video_jobs (
      id BIGSERIAL PRIMARY KEY,
      source TEXT NOT NULL,
      source_type VARCHAR(24) NOT NULL DEFAULT 'url',
      status VARCHAR(24) NOT NULL DEFAULT 'queued',
      processed_frames INTEGER NOT NULL DEFAULT 0,
      total_frames INTEGER,
      violation_count INTEGER NOT NULL DEFAULT 0,
      output_video_path TEXT,
      preview_frame_path TEXT,
      error_message TEXT,
      started_at TIMESTAMPTZ,
      finished_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      CONSTRAINT video_jobs_status_check CHECK (status IN ('queued', 'running', 'paused', 'completed', 'failed', 'canceled'))
    );
  `);

  await pool.query("ALTER TABLE video_jobs DROP CONSTRAINT IF EXISTS video_jobs_status_check;");
  await pool.query(`
    ALTER TABLE video_jobs
    ADD CONSTRAINT video_jobs_status_check
    CHECK (status IN ('queued', 'running', 'paused', 'completed', 'failed', 'canceled'));
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS violations (
      id BIGSERIAL PRIMARY KEY,
      video_job_id BIGINT REFERENCES video_jobs(id) ON DELETE SET NULL,
      violation_type VARCHAR(120) NOT NULL,
      detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      photo_path TEXT,
      source VARCHAR(80) NOT NULL DEFAULT 'opencv',
      helmet_detected BOOLEAN NOT NULL DEFAULT FALSE,
      vest_detected BOOLEAN NOT NULL DEFAULT FALSE,
      danger_zone BOOLEAN NOT NULL DEFAULT FALSE,
      bbox JSONB,
      confidence NUMERIC(5, 4),
      status VARCHAR(24) NOT NULL DEFAULT 'open',
      note TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      CONSTRAINT violations_status_check CHECK (status IN ('open', 'resolved', 'false_alarm'))
    );
  `);

  await pool.query("ALTER TABLE violations ADD COLUMN IF NOT EXISTS video_job_id BIGINT REFERENCES video_jobs(id) ON DELETE SET NULL;");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_video_jobs_created_at ON video_jobs (created_at DESC);");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_video_jobs_status ON video_jobs (status);");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_violations_detected_at ON violations (detected_at DESC);");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_violations_status ON violations (status);");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_violations_type ON violations (violation_type);");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_violations_video_job_id ON violations (video_job_id);");
}

export function mapViolation(row) {
  return {
    id: Number(row.id),
    videoJobId: row.video_job_id === null ? null : Number(row.video_job_id),
    violationType: row.violation_type,
    detectedAt: row.detected_at,
    photoPath: row.photo_path,
    photoUrl: row.photo_path ? `/api/violations/${row.id}/photo` : null,
    source: row.source,
    helmetDetected: row.helmet_detected,
    vestDetected: row.vest_detected,
    dangerZone: row.danger_zone,
    bbox: row.bbox,
    confidence: row.confidence === null ? null : Number(row.confidence),
    status: row.status,
    note: row.note,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

export function mapVideoJob(row) {
  return {
    id: Number(row.id),
    source: row.source,
    sourceType: row.source_type,
    status: row.status,
    processedFrames: row.processed_frames,
    totalFrames: row.total_frames,
    violationCount: row.violation_count,
    outputVideoPath: row.output_video_path,
    outputVideoUrl: row.output_video_path ? `/api/video-jobs/${row.id}/video` : null,
    previewFramePath: row.preview_frame_path,
    previewFrameUrl: row.preview_frame_path ? `/api/video-jobs/${row.id}/preview` : null,
    errorMessage: row.error_message,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}
