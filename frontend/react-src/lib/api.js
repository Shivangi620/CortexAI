export function createApiClient() {
  return async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});

    const response = await fetch(path, { ...options, headers });
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : await response.text();

    if (!response.ok) {
      const message =
        typeof body === "string" ? body : body?.detail || body?.error || `Request failed: ${response.status}`;
      throw new Error(message);
    }
    return body;
  };
}
