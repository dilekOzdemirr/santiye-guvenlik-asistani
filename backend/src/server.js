import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import fs from "node:fs";
import morgan from "morgan";
import multer from "multer";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { initDb, mapVideoJob, mapViolation, pool } from "./db.js";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const backendRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(backendRoot, process.env.REPO_ROOT || "..");
const uploadRoot = path.resolve(backendRoot, "uploads", "violations");
const videoUploadRoot = path.resolve(backendRoot, "uploads", "videos");
const jobControlRoot = path.resolve(backendRoot, "uploads", "job-controls");
const violationImageDir = path.resolve(backendRoot, process.env.VIOLATION_IMAGE_DIR || "../ihlaller");
const allowedPhotoRoots = [uploadRoot, violationImageDir, videoUploadRoot].map((item) => path.resolve(item));
const allowedVideoRoots = [videoUploadRoot].map((item) => path.resolve(item));

fs.mkdirSync(uploadRoot, { recursive: true });
fs.mkdirSync(videoUploadRoot, { recursive: true });
fs.mkdirSync(jobControlRoot, { recursive: true });
fs.mkdirSync(violationImageDir, { recursive: true });

const app = express();
const port = Number(process.env.PORT || 4000);
const corsOrigin = process.env.CORS_ORIGIN || "http://localhost:5173";

app.use(cors({ origin: corsOrigin === "*" ? true : corsOrigin }));
app.use(express.json({ limit: "5mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan("dev"));

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadRoot),
  filename: (_req, file, cb) => {
    const safeName = file.originalname.replace(/[^a-zA-Z0-9_.-]/g, "_");
    cb(null, `${Date.now()}-${safeName}`);
  }
});

const upload = multer({ storage });

const videoStorage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, videoUploadRoot),
  filename: (_req, file, cb) => {
    const safeName = file.originalname.replace(/[^a-zA-Z0-9_.-]/g, "_");
    cb(null, `${Date.now()}-${safeName}`);
  }
});

const uploadVideo = multer({
  storage: videoStorage,
  limits: { fileSize: 300 * 1024 * 1024 }
});

const activeVideoJobs = new Map();
const canceledVideoJobs = new Set();

function parseBoolean(value, fallback = false) {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value === 1;
  return ["true", "1", "yes", "evet", "var"].includes(String(value).toLowerCase());
}

function parseJson(value) {
  if (!value) return null;
  if (typeof value === "object") return value;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function firstValue(body, keys, fallback = undefined) {
  for (const key of keys) {
    if (body[key] !== undefined && body[key] !== null && body[key] !== "") {
      return body[key];
    }
  }
  return fallback;
}

function normalizePayload(body, file) {
  const helmetDetected = parseBoolean(firstValue(body, ["helmetDetected", "helmet_detected", "baret_var_mi"]));
  const vestDetected = parseBoolean(firstValue(body, ["vestDetected", "vest_detected", "yelek_var_mi"]));
  const dangerZone = parseBoolean(firstValue(body, ["dangerZone", "danger_zone", "tehlikeli_bolge"]));

  const missing = [];
  if (!helmetDetected) missing.push("Baret Yok");
  if (!vestDetected) missing.push("Yelek Yok");
  if (dangerZone) missing.push("Tehlikeli Bolge");

  const violationType =
    firstValue(body, ["violationType", "violation_type", "ihlal_turu"]) ||
    (missing.length > 0 ? missing.join(", ") : "KKD Ihlali");

  const confidenceValue = firstValue(body, ["confidence", "guven_skoru"]);
  const confidence = confidenceValue === undefined ? null : Number(confidenceValue);
  const requestedStatus = firstValue(body, ["status", "durum"], "open");
  const videoJobIdValue = firstValue(body, ["videoJobId", "video_job_id", "jobId"]);
  const videoJobId = videoJobIdValue === undefined ? null : Number(videoJobIdValue);

  return {
    videoJobId: Number.isFinite(videoJobId) ? videoJobId : null,
    violationType,
    detectedAt: firstValue(body, ["detectedAt", "detected_at", "tarih_saat"], new Date().toISOString()),
    photoPath: file ? file.path : firstValue(body, ["photoPath", "photo_path", "fotograf_yolu"], null),
    source: firstValue(body, ["source", "kaynak"], "opencv"),
    helmetDetected,
    vestDetected,
    dangerZone,
    bbox: parseJson(firstValue(body, ["bbox", "box", "koordinatlar"])),
    confidence: Number.isFinite(confidence) ? confidence : null,
    status: ["open", "resolved", "false_alarm"].includes(requestedStatus) ? requestedStatus : "open",
    note: firstValue(body, ["note", "not"], null)
  };
}

function resolveAllowedPath(mediaPath, roots) {
  if (!mediaPath) return null;

  const absolutePath = path.isAbsolute(mediaPath)
    ? path.resolve(mediaPath)
    : path.resolve(repoRoot, mediaPath);

  const isAllowed = roots.some((root) => {
    const relative = path.relative(root, absolutePath);
    return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
  });

  return isAllowed ? absolutePath : null;
}

function resolvePhotoPath(photoPath) {
  return resolveAllowedPath(photoPath, allowedPhotoRoots);
}

function resolveVideoPath(videoPath) {
  return resolveAllowedPath(videoPath, allowedVideoRoots);
}

function safeUnlink(filePath, roots) {
  const resolved = resolveAllowedPath(filePath, roots);
  if (resolved && fs.existsSync(resolved)) {
    fs.unlinkSync(resolved);
  }
}

function safeUnlinkPattern(root, prefix) {
  const resolvedRoot = path.resolve(root);
  if (!fs.existsSync(resolvedRoot)) return;

  for (const item of fs.readdirSync(resolvedRoot)) {
    if (!item.startsWith(prefix)) continue;
    safeUnlink(path.join(resolvedRoot, item), [resolvedRoot]);
  }
}

function stopActiveVideoJob(jobId) {
  canceledVideoJobs.add(jobId);
  fs.rmSync(pauseFilePath(jobId), { force: true });
  const child = activeVideoJobs.get(jobId);
  if (child && !child.killed) {
    child.kill("SIGTERM");
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true });
    }
  }
}

function pauseFilePath(jobId) {
  return path.join(jobControlRoot, `job_${jobId}.pause`);
}

function parseJobEvent(line) {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

function startVideoJob(jobId, source) {
  const scriptPath = path.resolve(repoRoot, "analyze_video_job.py");
  const pauseFile = pauseFilePath(jobId);
  fs.rmSync(pauseFile, { force: true });
  const pythonCommand = process.env.PYTHON_COMMAND || "python";
  const args = [
    scriptPath,
    "--job-id",
    String(jobId),
    "--source",
    source,
    "--api-url",
    `http://localhost:${port}/api/violations`,
    "--output-dir",
    videoUploadRoot,
    "--snapshot-dir",
    violationImageDir,
    "--pause-file",
    pauseFile
  ];

  const child = spawn(pythonCommand, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      SAFETY_API_URL: `http://localhost:${port}/api/violations`
    },
    windowsHide: true
  });

  activeVideoJobs.set(Number(jobId), child);

  let stdoutBuffer = "";
  let stderrBuffer = "";

  pool.query(
    "UPDATE video_jobs SET status = 'running', started_at = NOW(), updated_at = NOW() WHERE id = $1",
    [jobId]
  ).catch(console.error);

  child.stdout.on("data", (chunk) => {
    stdoutBuffer += chunk.toString();
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() || "";

    for (const line of lines) {
      const event = parseJobEvent(line.trim());
      if (!event) continue;

      if (event.event === "progress") {
        pool.query(
          `
            UPDATE video_jobs
            SET status = CASE WHEN status = 'queued' THEN 'running' ELSE status END,
                processed_frames = $1,
                total_frames = $2,
                preview_frame_path = COALESCE($3, preview_frame_path),
                updated_at = NOW()
            WHERE id = $4
          `,
          [event.processedFrames || 0, event.totalFrames || null, event.previewFramePath || null, jobId]
        ).catch(console.error);
      }

      if (event.event === "paused") {
        pool.query(
          `
            UPDATE video_jobs
            SET status = 'paused',
                processed_frames = $1,
                total_frames = $2,
                preview_frame_path = COALESCE($3, preview_frame_path),
                updated_at = NOW()
            WHERE id = $4 AND status NOT IN ('completed', 'failed', 'canceled')
          `,
          [event.processedFrames || 0, event.totalFrames || null, event.previewFramePath || null, jobId]
        ).catch(console.error);
      }

      if (event.event === "completed") {
        pool.query(
          `
            UPDATE video_jobs
            SET processed_frames = $1,
                total_frames = $2,
                violation_count = $3,
                output_video_path = $4,
                preview_frame_path = COALESCE($5, preview_frame_path),
                updated_at = NOW()
            WHERE id = $6
          `,
          [
            event.processedFrames || 0,
            event.totalFrames || null,
            event.violationCount || 0,
            event.outputVideoPath || null,
            event.previewFramePath || null,
            jobId
          ]
        ).catch(console.error);
      }
    }
  });

  child.stderr.on("data", (chunk) => {
    stderrBuffer = `${stderrBuffer}${chunk.toString()}`.slice(-4000);
  });

  child.on("close", async (code) => {
    activeVideoJobs.delete(Number(jobId));
    fs.rmSync(pauseFilePath(jobId), { force: true });

    if (canceledVideoJobs.has(Number(jobId))) {
      canceledVideoJobs.delete(Number(jobId));
      return;
    }

    if (code === 0) {
      await pool.query(
        `
          UPDATE video_jobs
          SET status = 'completed',
              violation_count = (SELECT COUNT(*)::int FROM violations WHERE video_job_id = $1),
              finished_at = NOW(),
              updated_at = NOW()
          WHERE id = $1 AND status NOT IN ('failed', 'canceled')
        `,
        [jobId]
      );
      return;
    }

    await pool.query(
      `
        UPDATE video_jobs
        SET status = 'failed',
            error_message = $1,
            finished_at = NOW(),
            updated_at = NOW()
        WHERE id = $2
      `,
      [stderrBuffer || `Python analizi ${code} kodu ile kapandi`, jobId]
    );
  });
}

app.get("/api/health", async (_req, res, next) => {
  try {
    await pool.query("SELECT 1");
    res.json({ ok: true, service: "santiye-guvenlik-api", database: "connected" });
  } catch (error) {
    next(error);
  }
});

app.post("/api/video-jobs/:id/cancel", async (req, res, next) => {
  try {
    const jobId = Number(req.params.id);
    const result = await pool.query("SELECT * FROM video_jobs WHERE id = $1", [jobId]);

    if (result.rowCount === 0) {
      return res.status(404).json({ error: "Analiz bulunamadi" });
    }

    const job = result.rows[0];
    if (!["queued", "running", "paused"].includes(job.status)) {
      return res.status(400).json({ error: "Bu analiz artik iptal edilemez" });
    }

    stopActiveVideoJob(jobId);

    const updated = await pool.query(
      `
        UPDATE video_jobs
        SET status = 'canceled',
            error_message = 'Analiz kullanici tarafindan iptal edildi.',
            finished_at = NOW(),
            updated_at = NOW()
        WHERE id = $1
        RETURNING *
      `,
      [jobId]
    );

    res.json({ data: mapVideoJob(updated.rows[0]) });
  } catch (error) {
    next(error);
  }
});

app.post("/api/video-jobs/:id/pause", async (req, res, next) => {
  try {
    const jobId = Number(req.params.id);
    const result = await pool.query("SELECT * FROM video_jobs WHERE id = $1", [jobId]);

    if (result.rowCount === 0) {
      return res.status(404).json({ error: "Analiz bulunamadi" });
    }

    const job = result.rows[0];
    if (!["queued", "running"].includes(job.status)) {
      return res.status(400).json({ error: "Bu analiz duraklatilamaz" });
    }

    fs.writeFileSync(pauseFilePath(jobId), new Date().toISOString(), "utf8");

    const updated = await pool.query(
      `
        UPDATE video_jobs
        SET status = 'paused',
            updated_at = NOW()
        WHERE id = $1
        RETURNING *
      `,
      [jobId]
    );

    res.json({ data: mapVideoJob(updated.rows[0]) });
  } catch (error) {
    next(error);
  }
});

app.post("/api/video-jobs/:id/resume", async (req, res, next) => {
  try {
    const jobId = Number(req.params.id);
    const result = await pool.query("SELECT * FROM video_jobs WHERE id = $1", [jobId]);

    if (result.rowCount === 0) {
      return res.status(404).json({ error: "Analiz bulunamadi" });
    }

    const job = result.rows[0];
    if (job.status !== "paused") {
      return res.status(400).json({ error: "Bu analiz devam ettirilemez" });
    }

    fs.rmSync(pauseFilePath(jobId), { force: true });

    const updated = await pool.query(
      `
        UPDATE video_jobs
        SET status = 'running',
            updated_at = NOW()
        WHERE id = $1
        RETURNING *
      `,
      [jobId]
    );

    res.json({ data: mapVideoJob(updated.rows[0]) });
  } catch (error) {
    next(error);
  }
});

app.delete("/api/video-jobs/:id", async (req, res, next) => {
  try {
    const jobId = Number(req.params.id);
    const jobResult = await pool.query("SELECT * FROM video_jobs WHERE id = $1", [jobId]);

    if (jobResult.rowCount === 0) {
      return res.status(404).json({ error: "Analiz bulunamadi" });
    }

    const job = jobResult.rows[0];
    if (["queued", "running", "paused"].includes(job.status)) {
      stopActiveVideoJob(jobId);
    }

    const violationResult = await pool.query("SELECT photo_path FROM violations WHERE video_job_id = $1", [jobId]);

    for (const row of violationResult.rows) {
      safeUnlink(row.photo_path, allowedPhotoRoots);
    }

    safeUnlink(job.output_video_path, allowedVideoRoots);
    safeUnlink(job.preview_frame_path, allowedPhotoRoots);
    safeUnlink(job.source, allowedVideoRoots);
    safeUnlinkPattern(videoUploadRoot, `job_${jobId}_`);

    await pool.query("DELETE FROM violations WHERE video_job_id = $1", [jobId]);
    await pool.query("DELETE FROM video_jobs WHERE id = $1", [jobId]);

    res.json({ ok: true, deletedId: jobId });
  } catch (error) {
    next(error);
  }
});

app.get("/api/stats", async (_req, res, next) => {
  try {
    const [summary, byType, recent] = await Promise.all([
      pool.query(`
        SELECT
          COUNT(*)::int AS total,
          COUNT(*) FILTER (WHERE status = 'open')::int AS open,
          COUNT(*) FILTER (WHERE detected_at::date = CURRENT_DATE)::int AS today,
          COUNT(*) FILTER (WHERE danger_zone = TRUE)::int AS danger_zone
        FROM violations
      `),
      pool.query(`
        SELECT violation_type, COUNT(*)::int AS count
        FROM violations
        GROUP BY violation_type
        ORDER BY count DESC, violation_type ASC
        LIMIT 8
      `),
      pool.query(`
        SELECT *
        FROM violations
        ORDER BY detected_at DESC
        LIMIT 1
      `)
    ]);

    res.json({
      summary: summary.rows[0],
      byType: byType.rows.map((row) => ({ violationType: row.violation_type, count: row.count })),
      latest: recent.rows[0] ? mapViolation(recent.rows[0]) : null
    });
  } catch (error) {
    next(error);
  }
});

app.get("/api/violations", async (req, res, next) => {
  try {
    const filters = [];
    const values = [];
    const limit = Math.min(Number(req.query.limit || 50), 200);

    if (req.query.status && req.query.status !== "all") {
      values.push(req.query.status);
      filters.push(`status = $${values.length}`);
    }

    if (req.query.type && req.query.type !== "all") {
      values.push(req.query.type);
      filters.push(`violation_type = $${values.length}`);
    }

    if (req.query.videoJobId) {
      values.push(req.query.videoJobId);
      filters.push(`video_job_id = $${values.length}`);
    }

    values.push(limit);

    const result = await pool.query(
      `
        SELECT *
        FROM violations
        ${filters.length ? `WHERE ${filters.join(" AND ")}` : ""}
        ORDER BY detected_at DESC
        LIMIT $${values.length}
      `,
      values
    );

    res.json({ data: result.rows.map(mapViolation) });
  } catch (error) {
    next(error);
  }
});

app.post("/api/violations", upload.single("photo"), async (req, res, next) => {
  try {
    const payload = normalizePayload(req.body, req.file);

    const result = await pool.query(
      `
        INSERT INTO violations (
          video_job_id,
          violation_type,
          detected_at,
          photo_path,
          source,
          helmet_detected,
          vest_detected,
          danger_zone,
          bbox,
          confidence,
          status,
          note
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING *
      `,
      [
        payload.videoJobId,
        payload.violationType,
        payload.detectedAt,
        payload.photoPath,
        payload.source,
        payload.helmetDetected,
        payload.vestDetected,
        payload.dangerZone,
        payload.bbox,
        payload.confidence,
        payload.status,
        payload.note
      ]
    );

    res.status(201).json({ data: mapViolation(result.rows[0]) });
  } catch (error) {
    next(error);
  }
});

app.get("/api/video-jobs", async (_req, res, next) => {
  try {
    const result = await pool.query(`
      SELECT
        video_jobs.*,
        COUNT(violations.id)::int AS violation_count
      FROM video_jobs
      LEFT JOIN violations ON violations.video_job_id = video_jobs.id
      GROUP BY video_jobs.id
      ORDER BY video_jobs.created_at DESC
      LIMIT 20
    `);

    res.json({ data: result.rows.map(mapVideoJob) });
  } catch (error) {
    next(error);
  }
});

app.post("/api/video-jobs", uploadVideo.single("video"), async (req, res, next) => {
  try {
    const submittedSource = firstValue(req.body, ["videoUrl", "source", "url"]);
    const source = req.file ? req.file.path : submittedSource;
    const sourceType = req.file ? "upload" : "url";

    if (!source) {
      return res.status(400).json({ error: "Video kaynagi gerekli" });
    }

    const result = await pool.query(
      `
        INSERT INTO video_jobs (source, source_type, status)
        VALUES ($1, $2, 'queued')
        RETURNING *
      `,
      [source, sourceType]
    );

    const job = mapVideoJob(result.rows[0]);
    startVideoJob(job.id, source);
    res.status(201).json({ data: job });
  } catch (error) {
    next(error);
  }
});

app.get("/api/video-jobs/:id/video", async (req, res, next) => {
  try {
    const result = await pool.query("SELECT output_video_path FROM video_jobs WHERE id = $1", [req.params.id]);
    if (result.rowCount === 0) {
      return res.status(404).json({ error: "Analiz bulunamadi" });
    }

    const videoPath = resolveVideoPath(result.rows[0].output_video_path);
    if (!videoPath || !fs.existsSync(videoPath)) {
      return res.status(404).json({ error: "Video bulunamadi" });
    }

    res.sendFile(videoPath);
  } catch (error) {
    next(error);
  }
});

app.get("/api/video-jobs/:id/preview", async (req, res, next) => {
  try {
    const result = await pool.query("SELECT preview_frame_path FROM video_jobs WHERE id = $1", [req.params.id]);
    if (result.rowCount === 0) {
      return res.status(404).json({ error: "Analiz bulunamadi" });
    }

    const framePath = resolvePhotoPath(result.rows[0].preview_frame_path);
    if (!framePath || !fs.existsSync(framePath)) {
      return res.status(404).json({ error: "Onizleme bulunamadi" });
    }

    res.sendFile(framePath);
  } catch (error) {
    next(error);
  }
});

app.patch("/api/violations/:id/status", async (req, res, next) => {
  try {
    const status = firstValue(req.body, ["status", "durum"]);
    if (!["open", "resolved", "false_alarm"].includes(status)) {
      return res.status(400).json({ error: "Gecersiz durum" });
    }

    const result = await pool.query(
      `
        UPDATE violations
        SET status = $1, updated_at = NOW()
        WHERE id = $2
        RETURNING *
      `,
      [status, req.params.id]
    );

    if (result.rowCount === 0) {
      return res.status(404).json({ error: "Ihlal bulunamadi" });
    }

    res.json({ data: mapViolation(result.rows[0]) });
  } catch (error) {
    next(error);
  }
});

app.get("/api/violations/:id/photo", async (req, res, next) => {
  try {
    const result = await pool.query("SELECT photo_path FROM violations WHERE id = $1", [req.params.id]);
    if (result.rowCount === 0) {
      return res.status(404).json({ error: "Ihlal bulunamadi" });
    }

    const photoPath = resolvePhotoPath(result.rows[0].photo_path);
    if (!photoPath || !fs.existsSync(photoPath)) {
      return res.status(404).json({ error: "Fotograf bulunamadi" });
    }

    res.sendFile(photoPath);
  } catch (error) {
    next(error);
  }
});

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(500).json({ error: "Sunucu hatasi", detail: error.message });
});

initDb()
  .then(() => {
    app.listen(port, () => {
      console.log(`Santiye guvenlik API http://localhost:${port} adresinde calisiyor`);
    });
  })
  .catch((error) => {
    console.error("Veritabani baslatilamadi:", error);
    process.exit(1);
  });
