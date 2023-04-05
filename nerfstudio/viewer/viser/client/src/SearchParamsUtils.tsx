/** Utilities for interacting with the URL search parameters.
 *
 * This lets us specify the websocket server + port from the URL. */

const key = "server";

export function getServersFromSearchParams() {
  return new URLSearchParams(window.location.search).getAll(key);
}

export function syncSearchParamServer(panelKey: number, server: string) {
  // Add/update servers in the URL bar.
  let serverParams = getServersFromSearchParams();
  if (panelKey >= serverParams.length) {
    serverParams.push(server);
  } else {
    serverParams[panelKey] = server;
  }

  // No need to update the URL bar if the websocket port matches the HTTP port.
  // So if we navigate to http://localhost:8081, this should by default connect to ws://localhost:8081.
  if (
    panelKey === 0 &&
    window.location.host.includes(
      server.replace("ws://", "").replace("/", "")
    ) &&
    serverParams.length === 1
  )
    serverParams = [];

  window.history.replaceState(
    null,
    "Viser",
    // We could use URLSearchParams() to build this string, but that would escape
    // it. We're going to just not escape the string. :)
    serverParams.length === 0
      ? window.location.href.split("?")[0]
      : "?" + serverParams.map((s) => key + "=" + s).join("&")
  );
}

export function truncateSearchParamServers(length: number) {
  const serverParams = getServersFromSearchParams().slice(0, length);
  window.history.replaceState(
    null,
    "Viser",
    // We could URLSearchParams() to build this string, but that would escape
    // it. We're going to just not escape the string. :)
    "?" + serverParams.map((s) => key + "=" + s).join("&")
  );
}
