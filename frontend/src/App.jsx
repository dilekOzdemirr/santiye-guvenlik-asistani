import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  Clock3,
  Database,
  Filter,
  ImageOff,
  RefreshCw,
  ShieldAlert,
  Siren,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import VideoAnalysis from "./VideoAnalysis.jsx";

const API_BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:4000").replace(/\/$/, "");

const statusOptions = [
  { value: "all", label: "Tümü" },
  { value: "open", label: "Açık" },
  { value: "resolved", label: "Çözüldü" },
  { value: "false_alarm", label: "Yanlış Alarm" }
];

const statusMeta = {
  open: { label: "Açık", className: "bg-red-50 text-danger border-red-200", icon: Siren },
  resolved: { label: "Çözüldü", className: "bg-emerald-50 text-mint border-emerald-200", icon: CheckCircle2 },
  false_alarm: { label: "Yanlış Alarm", className: "bg-zinc-100 text-steel border-zinc-200", icon: XCircle }
};

function apiUrl(path) {
  if (!path) return null;
  return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
}

function formatDate(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "short",
    timeStyle: "medium"
  }).format(new Date(value));
}

function MetricCard({ icon: Icon, label, value, tone }) {
  return (
    <section className={`rounded-lg border bg-white p-4 shadow-soft ${tone}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-steel">{label}</span>
        <Icon className="h-5 w-5" aria-hidden="true" />
      </div>
      <div className="mt-3 text-3xl font-bold tracking-normal text-ink">{value ?? 0}</div>
    </section>
  );
}

function StatusPill({ status }) {
  const meta = statusMeta[status] || statusMeta.open;
  const Icon = meta.icon;

  return (
    <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs font-semibold ${meta.className}`}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {meta.label}
    </span>
  );
}

function PhotoThumb({ violation, onSelect }) {
  if (!violation.photoUrl) {
    return (
      <div className="flex h-14 w-20 items-center justify-center rounded-md border border-zinc-200 bg-zinc-100 text-steel">
        <ImageOff className="h-5 w-5" aria-hidden="true" />
      </div>
    );
  }

  return (
    <button
      type="button"
      className="h-14 w-20 overflow-hidden rounded-md border border-zinc-200 bg-zinc-100"
      onClick={() => onSelect(violation)}
      title="Fotoğrafı büyüt"
    >
      <img
        src={apiUrl(violation.photoUrl)}
        alt={`${violation.violationType} ihlal görüntüsü`}
        className="h-full w-full object-cover"
      />
    </button>
  );
}

export default function App() {
  const [violations, setViolations] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedVideoJobId, setSelectedVideoJobId] = useState(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const loadData = useCallback(async () => {
    try {
      const query = new URLSearchParams({ limit: selectedVideoJobId ? "200" : "80" });
      if (statusFilter !== "all") query.set("status", statusFilter);
      if (typeFilter !== "all") query.set("type", typeFilter);
      if (selectedVideoJobId) query.set("videoJobId", String(selectedVideoJobId));

      const [violationsResponse, statsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/violations?${query.toString()}`),
        fetch(`${API_BASE_URL}/api/stats`)
      ]);

      if (!violationsResponse.ok || !statsResponse.ok) {
        throw new Error("API yanıt vermedi");
      }

      const violationsJson = await violationsResponse.json();
      const statsJson = await statsResponse.json();

      setViolations(violationsJson.data || []);
      setStats(statsJson);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [selectedVideoJobId, statusFilter, typeFilter]);

  useEffect(() => {
    loadData();
    const timer = window.setInterval(loadData, 5000);
    return () => window.clearInterval(timer);
  }, [loadData]);

  const typeOptions = useMemo(() => {
    const fromStats = stats?.byType?.map((item) => item.violationType) || [];
    const fromRows = violations.map((item) => item.violationType);
    return ["all", ...new Set([...fromStats, ...fromRows])];
  }, [stats, violations]);

  async function updateStatus(id, status) {
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/violations/${id}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status })
      });

      if (!response.ok) throw new Error("Durum güncellenemedi");
      await loadData();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSaving(false);
    }
  }

  const summary = stats?.summary || {};
  const latest = stats?.latest;

  return (
    <main className="min-h-screen bg-panel">
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div>
            <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-normal text-warning">
              <ShieldAlert className="h-5 w-5" aria-hidden="true" />
              Akıllı Şantiye Güvenlik Asistanı
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-normal text-ink sm:text-3xl">Yönetim Paneli</h1>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={loadData}
              className="inline-flex items-center gap-2 rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-semibold text-ink hover:bg-zinc-50"
              title="Verileri yenile"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Yenile
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {error && (
          <div className="mb-5 flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-danger">
            <AlertTriangle className="h-5 w-5 shrink-0" aria-hidden="true" />
            {error}
          </div>
        )}

        {latest && latest.status === "open" && (
          <section className="mb-5 rounded-lg border border-red-200 bg-white px-4 py-3 shadow-soft">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-md bg-red-50 text-danger">
                  <Siren className="h-5 w-5" aria-hidden="true" />
                </span>
                <div>
                  <p className="text-sm font-bold text-danger">Son açık ihlal</p>
                  <p className="text-sm text-steel">{latest.violationType} · {formatDate(latest.detectedAt)}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelected(latest)}
                className="inline-flex items-center gap-2 rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-semibold text-ink hover:bg-zinc-50"
              >
                <Camera className="h-4 w-4" aria-hidden="true" />
                Görüntüle
              </button>
            </div>
          </section>
        )}

        <VideoAnalysis
          apiBaseUrl={API_BASE_URL}
          onJobChange={loadData}
          onSelectedJobChange={setSelectedVideoJobId}
        />

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard icon={Database} label="Toplam İhlal" value={summary.total} tone="border-zinc-200 text-steel" />
          <MetricCard icon={Siren} label="Açık İhlal" value={summary.open} tone="border-red-100 text-danger" />
          <MetricCard icon={Clock3} label="Bugün" value={summary.today} tone="border-amber-100 text-warning" />
          <MetricCard icon={Activity} label="Tehlikeli Bölge" value={summary.danger_zone} tone="border-teal-100 text-mint" />
        </div>

        <section className="mt-6 rounded-lg border border-zinc-200 bg-white shadow-soft">
          <div className="flex flex-col gap-3 border-b border-zinc-200 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-2">
              <Filter className="h-5 w-5 text-steel" aria-hidden="true" />
              <h2 className="text-lg font-bold tracking-normal text-ink">
                {selectedVideoJobId ? `Analiz #${selectedVideoJobId} İhlalleri` : "İhlal Kayıtları"}
              </h2>
            </div>

            <div className="flex flex-col gap-2 sm:flex-row">
              <select
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value)}
                className="h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm font-semibold text-ink"
                aria-label="Durum filtresi"
              >
                {statusOptions.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
              <select
                value={typeFilter}
                onChange={(event) => setTypeFilter(event.target.value)}
                className="h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm font-semibold text-ink"
                aria-label="İhlal türü filtresi"
              >
                {typeOptions.map((type) => (
                  <option key={type} value={type}>{type === "all" ? "Tüm Türler" : type}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="overflow-x-auto scrollbar-thin">
            <table className="min-w-full divide-y divide-zinc-200 text-left text-sm">
              <thead className="bg-zinc-50 text-xs font-bold uppercase tracking-normal text-steel">
                <tr>
                  <th className="px-4 py-3">Fotoğraf</th>
                  <th className="px-4 py-3">İhlal Türü</th>
                  <th className="px-4 py-3">Zaman</th>
                  <th className="px-4 py-3">Kaynak</th>
                  <th className="px-4 py-3">Durum</th>
                  <th className="px-4 py-3 text-right">İşlem</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {loading ? (
                  <tr>
                    <td colSpan="6" className="px-4 py-10 text-center font-semibold text-steel">Yükleniyor...</td>
                  </tr>
                ) : violations.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="px-4 py-10 text-center font-semibold text-steel">Kayıt bulunamadı</td>
                  </tr>
                ) : (
                  violations.map((violation) => (
                    <tr key={violation.id} className="align-middle hover:bg-zinc-50">
                      <td className="px-4 py-3">
                        <PhotoThumb violation={violation} onSelect={setSelected} />
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-bold text-ink">{violation.violationType}</div>
                        <div className="mt-1 text-xs text-steel">
                          Baret: {violation.helmetDetected ? "Var" : "Yok"} · Yelek: {violation.vestDetected ? "Var" : "Yok"}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-steel">{formatDate(violation.detectedAt)}</td>
                      <td className="px-4 py-3 text-steel">{violation.source}</td>
                      <td className="px-4 py-3"><StatusPill status={violation.status} /></td>
                      <td className="px-4 py-3">
                        <div className="flex justify-end gap-2">
                          <button
                            type="button"
                            onClick={() => updateStatus(violation.id, "resolved")}
                            disabled={saving || violation.status === "resolved"}
                            className="rounded-md border border-emerald-200 px-2 py-1 text-xs font-bold text-mint hover:bg-emerald-50 disabled:opacity-40"
                          >
                            Çöz
                          </button>
                          <button
                            type="button"
                            onClick={() => updateStatus(violation.id, "false_alarm")}
                            disabled={saving || violation.status === "false_alarm"}
                            className="rounded-md border border-zinc-300 px-2 py-1 text-xs font-bold text-steel hover:bg-zinc-50 disabled:opacity-40"
                          >
                            Alarm Değil
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {selected && (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true">
          <section className="w-full max-w-3xl rounded-lg bg-white shadow-soft">
            <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
              <div>
                <h2 className="text-lg font-bold text-ink">{selected.violationType}</h2>
                <p className="text-sm text-steel">{formatDate(selected.detectedAt)}</p>
              </div>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="rounded-md border border-zinc-300 p-2 text-steel hover:bg-zinc-50"
                title="Kapat"
              >
                <XCircle className="h-5 w-5" aria-hidden="true" />
              </button>
            </div>
            <div className="p-4">
              {selected.photoUrl ? (
                <img
                  src={apiUrl(selected.photoUrl)}
                  alt={`${selected.violationType} detay görüntüsü`}
                  className="max-h-[70vh] w-full rounded-md object-contain"
                />
              ) : (
                <div className="flex h-80 items-center justify-center rounded-md border border-zinc-200 bg-zinc-100 text-steel">
                  <ImageOff className="h-8 w-8" aria-hidden="true" />
                </div>
              )}
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
