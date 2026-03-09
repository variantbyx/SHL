"use client";

import { useState } from "react";
import { Loader2, Zap, CheckCircle2, AlertCircle } from "lucide-react";

type EngineStatus = "idle" | "loading" | "ready" | "error";

function getApiBase(): string {
  const env = process.env.NEXT_PUBLIC_API_URL as string | undefined;
  if (env) return env.replace(/\/$/, "");
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host && host !== "localhost" && host !== "127.0.0.1") {
      return "https://shl-recommender-engine.onrender.com";
    }
  }
  return "http://localhost:8000";
}

export default function EngineStarter({ onReady }: { onReady?: () => void }) {
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");

  const startEngine = async () => {
    setStatus("loading");
    setErrorMsg("");
    try {
      const apiBase = getApiBase();
      const controller = new AbortController();
      // Give the cold-start up to 120 seconds
      const timeoutId = setTimeout(() => controller.abort(), 120_000);
      const res = await fetch(`${apiBase}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok)
        throw new Error(`Backend responded with status ${res.status}`);
      setStatus("ready");
      onReady?.();
    } catch (err: unknown) {
      const isAbort =
        err &&
        typeof err === "object" &&
        "name" in err &&
        (err as { name: string }).name === "AbortError";
      setErrorMsg(
        isAbort
          ? "Engine took too long to respond (>120 s). Please try again."
          : err instanceof Error
            ? err.message
            : "Unknown error starting engine.",
      );
      setStatus("error");
    }
  };

  if (status === "ready") {
    return (
      <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-green-50 border border-green-200 text-green-700 font-medium shadow-sm">
        <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-green-500" />
        Engine started! You can now proceed.
      </div>
    );
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <button
        onClick={startEngine}
        disabled={status === "loading"}
        className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-orange-500 hover:bg-orange-600 active:bg-orange-700 disabled:opacity-60 text-white font-semibold shadow-md transition-all duration-200"
      >
        {status === "loading" ? (
          <>
            <Loader2 className="animate-spin h-5 w-5" />
            Starting engine…
          </>
        ) : (
          <>
            <Zap className="h-5 w-5" />
            Start the Engine
          </>
        )}
      </button>

      {status === "loading" && (
        <p className="text-sm text-gray-500 ml-1 animate-pulse">
          Warming up the backend on Render — this may take up to a minute…
        </p>
      )}

      {status === "error" && (
        <div className="flex items-start gap-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2 mt-1">
          <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
}
