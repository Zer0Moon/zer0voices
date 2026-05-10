import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = "ws://127.0.0.1:8765";

export type SocketMessage = {
  type: string;
  payload?: unknown;
};

export function useSocket() {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const listeners = useRef<Map<string, (payload: unknown) => void>>(new Map());

  useEffect(() => {
    function connect() {
      const socket = new WebSocket(WS_URL);

      socket.onopen = () => {
        console.log("[Zer0Voices] Connected to sidecar");
        setConnected(true);
        socket.send(JSON.stringify({ type: "ping" }));
      };

      socket.onmessage = (event) => {
        const data: SocketMessage = JSON.parse(event.data);
        const listener = listeners.current.get(data.type);
        if (listener) listener(data.payload);
      };

      socket.onclose = () => {
        console.log("[Zer0Voices] Disconnected — retrying in 2s");
        setConnected(false);
        setTimeout(connect, 2000);
      };

      socket.onerror = () => {
        socket.close();
      };

      ws.current = socket;
    }

    connect();

    return () => {
      ws.current?.close();
    };
  }, []);

  const send = useCallback((type: string, payload?: unknown) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type, payload }));
    }
  }, []);

  const on = useCallback((type: string, cb: (payload: unknown) => void) => {
    listeners.current.set(type, cb);
  }, []);

  return { connected, send, on };
}