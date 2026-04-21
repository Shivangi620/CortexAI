import { useEffect } from "react";

export function usePolling(callback, delay, enabled = true) {
  useEffect(() => {
    if (!enabled || !delay) return undefined;
    const id = window.setInterval(() => {
      callback();
    }, delay);
    return () => window.clearInterval(id);
  }, [callback, delay, enabled]);
}
