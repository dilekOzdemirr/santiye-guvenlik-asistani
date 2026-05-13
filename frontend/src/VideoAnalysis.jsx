import {
  AlertTriangle,
  CheckCircle2,
  FileVideo,
  ImageOff,
  Link,
  Loader2,
  Pause,
  Play,
  PlayCircle,
  Trash2,
  Upload,
  Video,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

const statusMeta = {
  queued: { label: "Sırada", className: "border-zinc-200 bg-zinc-100 text-steel" },
  running: { label: "Analiz Ediliyor", className: "border-amber-200 bg-amber-50 text-warning" },
  paused: { label: "Duraklatıldı", className: "border-sky-200 bg-sky-50 text-steel" },
  completed: { label: "Tamamlandı", className: "border-emerald-200 bg-emerald-50 text-mint" },
  failed: { label: "Hata", className: "border-red-200 bg-red-50 text-danger" },
  canceled: { label: "İptal Edildi", className: "border-zinc-300 bg-zinc-100 text-steel" }
};

function apiUrl(apiBaseUrl, path) {
  if (!path) return null;
  return path.startsWith("http") ? path : `${apiBaseUrl}${path}`;
}

function formatDate(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "short",
    timeStyle: "short"
  }).format(new Date(value));
}

function JobStatus({ status }) {
  const meta = statusMeta[status] || statusMeta.queued;

  return (
    <span className={`inline-flex items-center rounded-md border px-2 py-1 text-xs font-bold ${meta.className}`}>
      {meta.label}
    </span>
  );
}

function progressValue(job) {
  if (!job.totalFrames || job.totalFrames <= 0) return job.status === "completed" ? 100 : 12;
  return Math.min(100, Math.round((job.processedFrames / job.totalFrames) * 100));
}

function canDeleteJob(job) {
  return ["completed", "failed", "canceled"].includes(job.status);
}

function canPauseJob(job) {
  return ["queued", "running"].includes(job.status);
}

function canResumeJob(job) {
  return job.status === "paused";
}

function canCancelJob(job) {
  return ["queued", "running", "paused"].includes(job.status);
}

export default function VideoAnalysis({ apiBaseUrl, onJobChange, onSelectedJobChange }) {
  const [jobs, setJobs] = useState([]);
  const [videoUrl, setVideoUrl] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [jobViolations, setJobViolations] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const loadJobs = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/video-jobs`);
      if (!response.ok) throw new Error("Video analizleri alınamadı");
      const json = await response.json();
      setJobs(json.data || []);
      setError("");
      onJobChange?.();
    } catch (requestError) {
      setError(requestError.message);
    }
  }, [apiBaseUrl, onJobChange]);

  useEffect(() => {
    loadJobs();
    const timer = window.setInterval(loadJobs, 4000);
    return () => window.clearInterval(timer);
  }, [loadJobs]);

  useEffect(() => {
    if (jobs.length === 0) return;

    const selectedJobStillExists = jobs.some((job) => job.id === selectedJobId);
    if (!selectedJobId || !selectedJobStillExists) {
      const firstPlayableJob = jobs.find((job) => job.status === "completed" && job.outputVideoUrl);
      setSelectedJobId((firstPlayableJob || jobs[0]).id);
    }
  }, [jobs, selectedJobId]);

  const selectedJob = useMemo(
    () => jobs.find((job) => job.id === selectedJobId) || jobs[0] || null,
    [jobs, selectedJobId]
  );

  useEffect(() => {
    onSelectedJobChange?.(selectedJob?.id || null);
  }, [onSelectedJobChange, selectedJob?.id]);

  const loadJobViolations = useCallback(async () => {
    if (!selectedJob?.id) {
      setJobViolations([]);
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/api/violations?videoJobId=${selectedJob.id}&limit=200`);
      if (!response.ok) throw new Error("Video ihlalleri alınamadı");
      const json = await response.json();
      setJobViolations(json.data || []);
    } catch {
      setJobViolations([]);
    }
  }, [apiBaseUrl, selectedJob?.id]);

  useEffect(() => {
    loadJobViolations();
    const timer = window.setInterval(loadJobViolations, 3000);
    return () => window.clearInterval(timer);
  }, [loadJobViolations]);

  async function submitVideo(event) {
    event.preventDefault();
    setSubmitting(true);

    try {
      const formData = new FormData();
      if (videoFile) {
        formData.append("video", videoFile);
      } else {
        formData.append("videoUrl", videoUrl.trim());
      }

      const response = await fetch(`${apiBaseUrl}/api/video-jobs`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) throw new Error("Video analizi başlatılamadı");
      const json = await response.json();
      setSelectedJobId(json.data.id);
      setVideoUrl("");
      setVideoFile(null);
      await loadJobs();
      onJobChange?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function cancelJob(jobId) {
    setSubmitting(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/video-jobs/${jobId}/cancel`, {
        method: "POST"
      });

      if (!response.ok) throw new Error("Analiz iptal edilemedi");
      await loadJobs();
      onJobChange?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function pauseJob(jobId) {
    setSubmitting(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/video-jobs/${jobId}/pause`, {
        method: "POST"
      });

      if (!response.ok) throw new Error("Analiz duraklatılamadı");
      await loadJobs();
      onJobChange?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function resumeJob(jobId) {
    setSubmitting(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/video-jobs/${jobId}/resume`, {
        method: "POST"
      });

      if (!response.ok) throw new Error("Analiz devam ettirilemedi");
      await loadJobs();
      onJobChange?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function deleteJob(jobId) {
    const approved = window.confirm(`Analiz #${jobId} silinsin mi?`);
    if (!approved) return;

    setSubmitting(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/video-jobs/${jobId}`, {
        method: "DELETE"
      });

      if (!response.ok) throw new Error("Analiz silinemedi");
      if (selectedJobId === jobId) {
        setSelectedJobId(null);
      }
      await loadJobs();
      onJobChange?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSubmitting(false);
    }
  }

  const selectedVideoUrl = selectedJob?.outputVideoUrl
    ? `${apiUrl(apiBaseUrl, selectedJob.outputVideoUrl)}?v=${encodeURIComponent(selectedJob.updatedAt || selectedJob.finishedAt || selectedJob.id)}`
    : null;

  const selectedPreviewUrl = selectedJob?.previewFrameUrl
    ? `${apiUrl(apiBaseUrl, selectedJob.previewFrameUrl)}?v=${encodeURIComponent(selectedJob.updatedAt || selectedJob.id)}`
    : null;

  return (
    <section className="mb-6 rounded-lg border border-zinc-200 bg-white shadow-soft">
      <div className="flex flex-col gap-4 border-b border-zinc-200 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-2">
          <Video className="h-5 w-5 text-warning" aria-hidden="true" />
          <h2 className="text-lg font-bold tracking-normal text-ink">Video Analizi</h2>
        </div>

        <form onSubmit={submitVideo} className="flex w-full flex-col gap-2 lg:max-w-3xl lg:flex-row">
          <label className="relative flex h-10 flex-1 items-center rounded-md border border-zinc-300 bg-white">
            <Link className="ml-3 h-4 w-4 text-steel" aria-hidden="true" />
            <input
              value={videoUrl}
              onChange={(event) => setVideoUrl(event.target.value)}
              disabled={Boolean(videoFile)}
              placeholder="https://.../santiye-video.mp4"
              className="h-full min-w-0 flex-1 rounded-md border-0 bg-transparent px-3 text-sm text-ink outline-none placeholder:text-zinc-400 disabled:opacity-50"
              aria-label="Video linki"
            />
          </label>

          <label className="inline-flex h-10 cursor-pointer items-center justify-center gap-2 rounded-md border border-zinc-300 bg-white px-3 text-sm font-semibold text-ink hover:bg-zinc-50">
            <Upload className="h-4 w-4" aria-hidden="true" />
            {videoFile ? videoFile.name : "Dosya"}
            <input
              type="file"
              accept="video/*"
              className="sr-only"
              onChange={(event) => setVideoFile(event.target.files?.[0] || null)}
            />
          </label>

          <button
            type="submit"
            disabled={submitting || (!videoUrl.trim() && !videoFile)}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-warning px-4 text-sm font-bold text-ink hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Play className="h-4 w-4" aria-hidden="true" />}
            Başlat
          </button>
        </form>
      </div>

      {error && (
        <div className="mx-4 mt-4 flex items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm font-semibold text-danger">
          <AlertTriangle className="h-4 w-4" aria-hidden="true" />
          {error}
        </div>
      )}

      <div className="grid gap-4 p-4 xl:grid-cols-[360px_1fr]">
        <div className="space-y-3">
          {jobs.length === 0 ? (
            <div className="flex h-32 items-center justify-center rounded-lg border border-dashed border-zinc-300 text-sm font-semibold text-steel">
              Analiz bekleniyor
            </div>
          ) : (
            jobs.map((job) => (
              <div
                key={job.id}
                onClick={() => setSelectedJobId(job.id)}
                className={`w-full rounded-lg border p-3 text-left transition ${
                  selectedJob?.id === job.id ? "border-warning bg-amber-50/50" : "border-zinc-200 bg-white hover:bg-zinc-50"
                }`}
                role="button"
                tabIndex={0}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 font-bold text-ink">
                      <FileVideo className="h-4 w-4 shrink-0 text-steel" aria-hidden="true" />
                      <span className="truncate">Analiz #{job.id}</span>
                    </div>
                    <p className="mt-1 truncate text-xs text-steel">{job.sourceType === "upload" ? "Yüklenen video" : job.source}</p>
                  </div>
                  <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
                    <JobStatus status={job.status} />
                    {canPauseJob(job) && (
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          pauseJob(job.id);
                        }}
                        disabled={submitting}
                        className="inline-flex items-center gap-1 rounded-md border border-amber-200 bg-white px-2 py-1 text-xs font-bold text-warning hover:bg-amber-50 disabled:opacity-50"
                      >
                        <Pause className="h-3.5 w-3.5" aria-hidden="true" />
                        Duraklat
                      </button>
                    )}
                    {canResumeJob(job) && (
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          resumeJob(job.id);
                        }}
                        disabled={submitting}
                        className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-white px-2 py-1 text-xs font-bold text-mint hover:bg-emerald-50 disabled:opacity-50"
                      >
                        <PlayCircle className="h-3.5 w-3.5" aria-hidden="true" />
                        Devam Et
                      </button>
                    )}
                    {canCancelJob(job) && (
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          cancelJob(job.id);
                        }}
                        disabled={submitting}
                        className="rounded-md border border-red-200 bg-white px-2 py-1 text-xs font-bold text-danger hover:bg-red-50 disabled:opacity-50"
                      >
                        İptal
                      </button>
                    )}
                    {canDeleteJob(job) && (
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          deleteJob(job.id);
                        }}
                        disabled={submitting}
                        className="inline-flex items-center gap-1 rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-bold text-steel hover:bg-zinc-50 disabled:opacity-50"
                      >
                        <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                        Sil
                      </button>
                    )}
                  </div>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-zinc-100">
                  <div className="h-full rounded-full bg-warning" style={{ width: `${progressValue(job)}%` }} />
                </div>
                <div className="mt-2 flex items-center justify-between text-xs font-semibold text-steel">
                  <span>{job.processedFrames || 0} / {job.totalFrames || "?"} kare</span>
                  <span>{job.violationCount || 0} ihlal</span>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="min-h-[320px] rounded-lg border border-zinc-200 bg-zinc-50 p-3">
          {!selectedJob ? (
            <div className="flex h-full min-h-[300px] items-center justify-center text-steel">
              <ImageOff className="h-8 w-8" aria-hidden="true" />
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-ink">Analiz #{selectedJob.id}</h3>
                    <JobStatus status={selectedJob.status} />
                  </div>
                  <p className="mt-1 text-xs font-semibold text-steel">{formatDate(selectedJob.createdAt)}</p>
                </div>
                {canDeleteJob(selectedJob) && (
                  <div className="flex items-center gap-2">
                    {selectedJob.status === "completed" && (
                      <span className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-white px-2 py-1 text-xs font-bold text-mint">
                        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                        {selectedJob.violationCount || 0} kayıt
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={() => deleteJob(selectedJob.id)}
                      disabled={submitting}
                      className="inline-flex items-center gap-1 rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-bold text-steel hover:bg-zinc-50 disabled:opacity-50"
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                      Sil
                    </button>
                  </div>
                )}
                {(canPauseJob(selectedJob) || canResumeJob(selectedJob) || canCancelJob(selectedJob)) && (
                  <div className="flex flex-wrap items-center gap-2">
                    {canPauseJob(selectedJob) && (
                      <button
                        type="button"
                        onClick={() => pauseJob(selectedJob.id)}
                        disabled={submitting}
                        className="inline-flex items-center gap-1 rounded-md border border-amber-200 bg-white px-2 py-1 text-xs font-bold text-warning hover:bg-amber-50 disabled:opacity-50"
                      >
                        <Pause className="h-3.5 w-3.5" aria-hidden="true" />
                        Duraklat
                      </button>
                    )}
                    {canResumeJob(selectedJob) && (
                      <button
                        type="button"
                        onClick={() => resumeJob(selectedJob.id)}
                        disabled={submitting}
                        className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-white px-2 py-1 text-xs font-bold text-mint hover:bg-emerald-50 disabled:opacity-50"
                      >
                        <PlayCircle className="h-3.5 w-3.5" aria-hidden="true" />
                        Devam Et
                      </button>
                    )}
                    {canCancelJob(selectedJob) && (
                      <button
                        type="button"
                        onClick={() => cancelJob(selectedJob.id)}
                        disabled={submitting}
                        className="inline-flex items-center gap-1 rounded-md border border-red-200 bg-white px-2 py-1 text-xs font-bold text-danger hover:bg-red-50 disabled:opacity-50"
                      >
                        <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
                        İptal Et
                      </button>
                    )}
                  </div>
                )}
              </div>

              {selectedVideoUrl ? (
                <video
                  key={selectedVideoUrl}
                  controls
                  preload="metadata"
                  poster={selectedPreviewUrl || undefined}
                  className="aspect-video w-full rounded-md bg-black object-contain"
                  src={selectedVideoUrl}
                />
              ) : selectedPreviewUrl ? (
                <img
                  src={selectedPreviewUrl}
                  alt={`Analiz ${selectedJob.id} önizleme karesi`}
                  className="aspect-video w-full rounded-md bg-black object-contain"
                />
              ) : (
                <div className="flex aspect-video w-full items-center justify-center rounded-md bg-zinc-200 text-steel">
                  <Loader2 className="h-8 w-8 animate-spin" aria-hidden="true" />
                </div>
              )}

              {selectedJob.errorMessage && (
                <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm font-semibold text-danger">
                  {selectedJob.errorMessage}
                </div>
              )}

              {jobViolations.length > 0 && (
                <div className="rounded-lg border border-zinc-200 bg-white p-3">
                  <div className="mb-3 flex items-center justify-between gap-2">
                    <h4 className="text-sm font-bold text-ink">Bu Videodaki İhlaller</h4>
                    <span className="rounded-md bg-red-50 px-2 py-1 text-xs font-bold text-danger">
                      {selectedJob.violationCount || jobViolations.length} kayıt
                    </span>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {jobViolations.map((violation) => (
                      <a
                        key={violation.id}
                        href={apiUrl(apiBaseUrl, violation.photoUrl)}
                        target="_blank"
                        rel="noreferrer"
                        className="overflow-hidden rounded-md border border-zinc-200 bg-zinc-50 hover:border-warning"
                      >
                        {violation.photoUrl ? (
                          <img
                            src={apiUrl(apiBaseUrl, violation.photoUrl)}
                            alt={`${violation.violationType} görüntüsü`}
                            className="aspect-video w-full bg-zinc-200 object-cover"
                          />
                        ) : (
                          <div className="flex aspect-video w-full items-center justify-center bg-zinc-200 text-steel">
                            <ImageOff className="h-6 w-6" aria-hidden="true" />
                          </div>
                        )}
                        <div className="p-2">
                          <p className="truncate text-sm font-bold text-ink">{violation.violationType}</p>
                          <p className="mt-1 text-xs font-semibold text-steel">
                            Baret: {violation.helmetDetected ? "Var" : "Yok"} · Yelek: {violation.vestDetected ? "Var" : "Yok"}
                          </p>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
