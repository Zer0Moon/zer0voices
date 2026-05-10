import { useEffect, useState } from "react";
import { useSocket } from "./hooks/useSocket";
import { MainScreen } from "./components/main/MainScreen";

type Screen = "main" | "models" | "browse" | "soundboard" | "settings";

interface GpuInfo {
  available: boolean;
  name?: string;
  vram_gb?: number;
  cuda_version?: string;
}

function App() {
  const { connected, send, on } = useSocket();
  const [screen, setScreen] = useState<Screen>("main");
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);

  useEffect(() => {
    on("pong", () => {
      send("get_gpu_info");
      send("get_devices");
    });
    on("gpu_info", (payload) => {
      setGpuInfo(payload as GpuInfo);
    });
  }, [on, send]);

  const navItems: { id: Screen; emoji: string }[] = [
    { id: "main", emoji: "〰️" },
    { id: "models", emoji: "🧠" },
    { id: "browse", emoji: "🧭" },
    { id: "soundboard", emoji: "▶️" },
    { id: "settings", emoji: "⚙️" },
  ];

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-white select-none">
      <div className="flex items-center gap-3 px-4 py-2 bg-zinc-900 border-b border-zinc-800">
        <div className="w-6 h-6 rounded-md bg-blue-600 flex items-center justify-center text-xs font-bold">
          Z
        </div>
        <span className="text-sm font-medium flex-1">Zer0Voices</span>
        <div className={`flex items-center gap-2 text-xs font-medium px-3 py-1 rounded-full ${gpuInfo?.available ? "bg-green-950 text-green-400" : "bg-zinc-800 text-zinc-400"}`}>
          <div className={`w-1.5 h-1.5 rounded-full ${gpuInfo?.available ? "bg-green-400" : "bg-zinc-500"}`} />
          {gpuInfo?.available ? `${gpuInfo.name} · CUDA ${gpuInfo.cuda_version}` : connected ? "Detecting GPU..." : "Connecting..."}
        </div>
        <div className="flex items-center gap-1 ml-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setScreen(item.id)}
              className={`w-8 h-8 rounded-md text-sm transition-colors ${screen === item.id ? "bg-blue-900 text-blue-400" : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"}`}
            >
              {item.emoji}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-hidden p-3">
        {screen === "main" && <MainScreen />}
        {screen === "models" && (
          <div className="flex items-center justify-center h-full text-zinc-500">Model Manager coming soon</div>
        )}
        {screen === "browse" && (
          <div className="flex items-center justify-center h-full text-zinc-500">Browse Models coming soon</div>
        )}
        {screen === "soundboard" && (
          <div className="flex items-center justify-center h-full text-zinc-500">Soundboard coming soon</div>
        )}
        {screen === "settings" && (
          <div className="flex items-center justify-center h-full text-zinc-500">Settings coming soon</div>
        )}
      </div>

      <div className="flex items-center gap-2 px-4 py-1.5 bg-zinc-900 border-t border-zinc-800 text-xs text-zinc-500">
        <div className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-green-400" : "bg-red-400"}`} />
        {connected ? "Sidecar connected" : "Sidecar disconnected"}
      </div>
    </div>
  );
}

export default App;