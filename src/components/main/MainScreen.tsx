import { useEffect, useRef, useState, useCallback } from "react";
import { useSocket } from "../../hooks/useSocket";

interface Device {
  id: number;
  name: string;
}

interface AudioData {
  waveform: number[];
  rms: number;
  converting: boolean;
}

export function MainScreen() {
  const { send, on } = useSocket();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const waveformRef = useRef<number[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [inputDevices, setInputDevices] = useState<Device[]>([]);
  const [outputDevices, setOutputDevices] = useState<Device[]>([]);
  const [selectedInput, setSelectedInput] = useState<number | null>(null);
  const [selectedOutput, setSelectedOutput] = useState<number | null>(null);
  const [rms, setRms] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [perfMode, setPerfMode] = useState<"low" | "balanced" | "quality">("balanced");
  const [passthrough, setPassthrough] = useState(false);
  const [monitoring, setMonitoring] = useState(true);
  const [latency] = useState(38);

  useEffect(() => {
    on("devices", (payload) => {
      const p = payload as { inputs: Device[]; outputs: Device[] };
      setInputDevices(p.inputs);
      setOutputDevices(p.outputs);
      if (p.inputs.length > 0) setSelectedInput(p.inputs[0].id);
      if (p.outputs.length > 0) setSelectedOutput(p.outputs[0].id);
    });

    on("audio_data", (payload) => {
      const p = payload as AudioData;
      waveformRef.current = p.waveform;
      setRms(p.rms);
    });

    on("stream_started", () => setIsStreaming(true));
    on("stream_stopped", () => setIsStreaming(false));

    // Request devices on mount
    send("get_devices");
  }, [on, send]);

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const wave = waveformRef.current;
    if (wave.length === 0) {
      ctx.beginPath();
      ctx.strokeStyle = "#3f3f46";
      ctx.lineWidth = 1.5;
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.strokeStyle = isConverting ? "#22c55e" : "#3b82f6";
      ctx.lineWidth = 1.5;
      ctx.shadowBlur = 6;
      ctx.shadowColor = isConverting ? "#22c55e44" : "#3b82f644";
      wave.forEach((v, i) => {
        const x = (i / wave.length) * W;
        const y = H / 2 + v * H * 2.5;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    animFrameRef.current = requestAnimationFrame(drawWaveform);
  }, [isConverting]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(drawWaveform);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [drawWaveform]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth * devicePixelRatio;
      canvas.height = canvas.offsetHeight * devicePixelRatio;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  const toggleStream = () => {
    if (isStreaming) {
      send("stop_stream");
      setIsConverting(false);
    } else {
      if (selectedInput !== null) send("set_input_device", selectedInput);
      if (selectedOutput !== null) send("set_output_device", selectedOutput);
      send("start_stream");
      setIsConverting(true);
    }
  };

  return (
    <div className="grid grid-cols-[1fr_260px] gap-3 h-full">

      {/* Left Column */}
      <div className="flex flex-col gap-3">

        {/* Waveform */}
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-zinc-500 tracking-wider">LIVE WAVEFORM</span>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
                <span className="text-zinc-400">Input</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
                <span className="text-zinc-400">Output</span>
              </span>
            </div>
          </div>
          <canvas
            ref={canvasRef}
            className="w-full h-20 block"
          />
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: "LATENCY", value: latency, unit: "ms", color: "text-green-400" },
            { label: "INPUT LEVEL", value: Math.round(rms * 1000), unit: "", color: "text-white" },
            { label: "SAMPLE RATE", value: "44.1", unit: "k", color: "text-white" },
          ].map((s) => (
            <div key={s.label} className="bg-zinc-900 rounded-xl border border-zinc-800 p-3">
              <div className="text-xs text-zinc-500 font-medium tracking-wider mb-1">{s.label}</div>
              <div className={`text-2xl font-medium ${s.color}`}>
                {s.value}
                <span className="text-sm font-normal text-zinc-500">{s.unit}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Pitch */}
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-white">Pitch shift</span>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-blue-400 min-w-[48px] text-right">
                {pitch > 0 ? `+${pitch}` : pitch} st
              </span>
              <button
                onClick={() => setPitch(0)}
                className="text-xs px-2 py-1 rounded-md border border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500 transition-colors"
              >
                Reset
              </button>
              <button className="text-xs px-3 py-1 rounded-md bg-blue-900 text-blue-400 border border-blue-800 hover:bg-blue-800 transition-colors">
                ✨ Auto-detect
              </button>
            </div>
          </div>
          <input
            type="range"
            min={-24}
            max={24}
            value={pitch}
            onChange={(e) => setPitch(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
          <div className="flex justify-between mt-1 text-xs text-zinc-600">
            <span>−24 st</span>
            <span>0</span>
            <span>+24 st</span>
          </div>
        </div>

        {/* Devices */}
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-zinc-500 font-medium tracking-wider mb-2">INPUT DEVICE</div>
              <select
                value={selectedInput ?? ""}
                onChange={(e) => setSelectedInput(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-600"
              >
                {inputDevices.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>
            <div>
              <div className="text-xs text-zinc-500 font-medium tracking-wider mb-2">OUTPUT DEVICE</div>
              <select
                value={selectedOutput ?? ""}
                onChange={(e) => setSelectedOutput(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-600"
              >
                {outputDevices.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

      </div>

      {/* Right Column */}
      <div className="flex flex-col gap-3">

        {/* Big Mic Button */}
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 flex flex-col items-center gap-3">
          <button
            onClick={toggleStream}
            className={`w-20 h-20 rounded-full flex items-center justify-center text-4xl transition-all border-2 ${
              isStreaming
                ? "bg-blue-950 border-blue-500 shadow-lg shadow-blue-500/20 scale-105"
                : "bg-zinc-800 border-zinc-600 hover:border-zinc-400"
            }`}
          >
            🎤
          </button>
          <span className={`text-sm font-medium ${isStreaming ? "text-blue-400" : "text-zinc-500"}`}>
            {isStreaming ? "Converting · active" : "Stopped"}
          </span>
        </div>

        {/* Model Card */}
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
          <div className="text-xs text-zinc-500 font-medium tracking-wider mb-3">LOADED MODEL</div>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-9 h-9 rounded-lg bg-blue-950 flex items-center justify-center text-lg flex-shrink-0">
              🧠
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-white truncate">No model loaded</div>
              <div className="text-xs text-zinc-500">Go to Model Manager</div>
            </div>
          </div>
          <button className="w-full text-xs py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500 transition-colors">
            Load a model →
          </button>
        </div>

        {/* Toggles */}
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4 flex flex-col gap-4">
          {[
            { label: "Passthrough", sub: "Bypass AI", value: passthrough, set: setPassthrough },
            { label: "Input monitoring", sub: "Hear yourself live", value: monitoring, set: setMonitoring },
          ].map((t) => (
            <div key={t.label} className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-white">{t.label}</div>
                <div className="text-xs text-zinc-500">{t.sub}</div>
              </div>
              <button
                onClick={() => t.set(!t.value)}
                className={`w-10 h-6 rounded-full transition-colors relative flex-shrink-0 ${t.value ? "bg-green-600" : "bg-zinc-700"}`}
              >
                <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${t.value ? "left-5" : "left-1"}`} />
              </button>
            </div>
          ))}
        </div>

        {/* Performance Mode */}
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
          <div className="text-xs text-zinc-500 font-medium tracking-wider mb-3">PERFORMANCE MODE</div>
          <div className="grid grid-cols-3 gap-2">
            {(["low", "balanced", "quality"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setPerfMode(mode)}
                className={`py-2 text-xs rounded-lg border capitalize transition-colors ${
                  perfMode === mode
                    ? "bg-blue-900 text-blue-400 border-blue-800"
                    : "bg-zinc-800 text-zinc-400 border-zinc-700 hover:border-zinc-500"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}