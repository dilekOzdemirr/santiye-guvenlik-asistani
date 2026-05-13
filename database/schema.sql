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

CREATE INDEX IF NOT EXISTS idx_video_jobs_created_at ON video_jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_jobs_status ON video_jobs (status);
CREATE INDEX IF NOT EXISTS idx_violations_detected_at ON violations (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_violations_status ON violations (status);
CREATE INDEX IF NOT EXISTS idx_violations_type ON violations (violation_type);
CREATE INDEX IF NOT EXISTS idx_violations_video_job_id ON violations (video_job_id);
