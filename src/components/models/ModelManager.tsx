import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useSocket } from "../../hooks/useSocket";

interface ModelInfo {
  path: string;
  name: string;
  version: string;
  sample_rate: string;
  f0: number;
  size_mb: number;
}

export function ModelManager() {
  const { send, on } = useSocket();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadedModel, setLoadedModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("zer0voices_models");
    if (saved) setModels(JSON.parse(saved));

    on("model_loaded", (payload) => {
      const info = payload as ModelInfo;
      setLoadedModel(info);
      setLoading(false);
      setError(null);
      setModels((prev) => {
        const exists = prev.find((m) => m.path === info.path);
        const updated = exists ? prev : [...prev, info];
        localStorage.setItem("zer0voices_models", JSON.stringify(updated));
        return updated;
      });
    });

    on("model_loading", () => {
      setLoading(true);
      setError(null);
    });

    on("model_error", (payload) => {
      setError(payload as string);
      setLoading(false);
    });
  }, [on]);

  const browseForModel = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "RVC Model", extensions: ["pth"] }],
      });
      if (selected) {
        send("load_model", selected);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const loadModel = (model: ModelInfo) => {
    send("load_model", model.path);
  };

  const removeModel = (path: string) => {
    setModels((prev) => {
      const updated = prev.filter((m) => m.path !== path);
      localStorage.setItem("zer0voices_models", JSON.stringify(updated));
      return updated;
    });
  };

  return (
    <div className="grid grid-cols-[1fr_280px] gap-3 h-full">

      {/* Left — Model List */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-base font-medium text-white">My models</span>
          <button
            onClick={browseForModel}
            className="text-xs px-3 py-2 rounded-lg bg-blue-900 text-blue-400 border border-blue-800 hover:bg-blue-800 transition-colors flex items-center gap-2"
          >
            + Browse for .pth
          </button>
        </div>

        {error && (
          <div className="bg-red-950 border border-red-800 rounded-xl p-3 text-sm text-red-400">
            ⚠️ {error}
          </div>
        )}

        {loading && (
          <div className="bg-blue-950 border border-blue-800 rounded-xl p-3 text-sm text-blue-400 flex items-center gap-2">
            <span className="animate-spin">⚙️</span> Loading model...
          </div>
        )}

        <div className="flex flex-col gap-2 overflow-y-auto">
          {models.length === 0 ? (
            <div
              onClick={browseForModel}
              className="border-2 border-dashed border-zinc-700 rounded-xl p-8 flex flex-col items-center gap-3 cursor-pointer hover:border-zinc-500 transition-colors"
            >
              <span className="text-3xl">📂</span>
              <span className="text-sm font-medium text-zinc-400">No models yet</span>
              <span className="text-xs text-zinc-600">Click to browse for a .pth file</span>
            </div>
          ) : (
            models.map((model) => (
              <div
                key={model.path}
                className={`bg-zinc-900 rounded-xl border p-4 flex items-center gap-3 transition-colors ${
                  loadedModel?.path === model.path
                    ? "border-blue-700"
                    : "border-zinc-800 hover:border-zinc-700"
                }`}
              >
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-xl flex-shrink-0 ${
                  loadedModel?.path === model.path ? "bg-blue-950" : "bg-zinc-800"
                }`}>
                  🧠
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-white truncate">{model.name}</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-blue-950 text-blue-400 flex-shrink-0">
                      {model.version}
                    </span>
                    {loadedModel?.path === model.path && (
                      <span className="text-xs px-2 py-0.5 rounded bg-green-950 text-green-400 flex-shrink-0">
                        Active
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-zinc-500">
                    {model.sample_rate} · {model.size_mb} MB · {model.f0 ? "f0 enabled" : "no f0"}
                  </div>
                </div>
                <div className="flex gap-2 flex-shrink-0">
                  {loadedModel?.path !== model.path && (
                    <button
                      onClick={() => loadModel(model)}
                      className="text-xs px-3 py-1.5 rounded-lg bg-blue-900 text-blue-400 border border-blue-800 hover:bg-blue-800 transition-colors"
                    >
                      Load
                    </button>
                  )}
                  <button
                    onClick={() => removeModel(model.path)}
                    className="text-xs px-3 py-1.5 rounded-lg bg-zinc-800 text-red-400 border border-zinc-700 hover:border-red-800 hover:bg-red-950 transition-colors"
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right — Model Details */}
      <div className="flex flex-col gap-3">
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
          <div className="text-xs text-zinc-500 font-medium tracking-wider mb-4">MODEL DETAILS</div>
          {loadedModel ? (
            <>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-11 h-11 rounded-xl bg-blue-950 flex items-center justify-center text-2xl">
                  🧠
                </div>
                <div>
                  <div className="text-sm font-medium text-white">{loadedModel.name}</div>
                  <div className="text-xs text-green-400">Loaded · active</div>
                </div>
              </div>
              <div className="flex flex-col gap-2 text-sm border-t border-zinc-800 pt-4">
                {[
                  { label: "Version", value: loadedModel.version },
                  { label: "Sample rate", value: loadedModel.sample_rate },
                  { label: "File size", value: `${loadedModel.size_mb} MB` },
                  { label: "Pitch (f0)", value: loadedModel.f0 ? "Enabled" : "Disabled" },
                ].map((row) => (
                  <div key={row.label} className="flex justify-between">
                    <span className="text-zinc-500">{row.label}</span>
                    <span className="text-white font-medium">{row.value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 py-6 text-center">
              <span className="text-3xl">🧠</span>
              <span className="text-sm text-zinc-500">No model loaded</span>
              <span className="text-xs text-zinc-600">Browse for a .pth file to get started</span>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}