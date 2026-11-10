"""Interfaccia a riga di comando.

Funziona in due modalita', scelte automaticamente: se il servizio e' in ascolto
usa la sua API (e quindi vede lo storico completo e la riserva attiva); se non
c'e', crea un'istanza locale usa-e-getta. Cosi' i comandi di sola lettura
funzionano sempre, anche prima di aver installato il servizio.

Le operazioni distruttive richiedono `--yes`, che e' l'equivalente testuale della
conferma richiesta dalla GUI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx

from ..core.config import Config, get_or_create_token
from ..core.units import human_bytes, parse_size, pct

BAR_WIDTH = 34


class Client:
    """Parla con il servizio se c'e', altrimenti lavora in proprio."""

    def __init__(self, cfg: Config, force_local: bool = False) -> None:
        self.cfg = cfg
        self.base = f"http://{cfg.api_host}:{cfg.api_port}"
        self.token = get_or_create_token() if cfg.api_require_token else None
        self._service = None
        self.remote = False if force_local else self._probe()

    def _probe(self) -> bool:
        try:
            r = httpx.get(f"{self.base}/health", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def service(self):
        if self._service is None:
            from ..service import VramService
            self._service = VramService(self.cfg)
        return self._service

    def get(self, path: str, **params: Any) -> Any:
        if not self.remote:
            raise RuntimeError("servizio non in esecuzione")
        r = httpx.get(f"{self.base}{path}", params=params,
                      headers={"x-vramexp-token": self.token or ""}, timeout=30.0)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: Any) -> Any:
        if not self.remote:
            raise RuntimeError("servizio non in esecuzione")
        r = httpx.post(f"{self.base}{path}", json=payload,
                       headers={"x-vramexp-token": self.token or ""}, timeout=60.0)
        return r.json()

    def close(self) -> None:
        if self._service is not None:
            self._service.stop()


# --- rendering -------------------------------------------------------------
def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    filled = int(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def print_status(status: dict, verbose: bool = False) -> None:
    for g in status.get("gpus", []):
        used_pct = g.get("used_pct", 0.0)
        print(f"\n  {g.get('name', 'GPU')}")
        print(f"  {bar(used_pct / 100)} {used_pct:5.1f}%   "
              f"{human_bytes(g['used_bytes'])} / {human_bytes(g['total_bytes'])} residenti")
        committed = g.get("committed_bytes")
        if committed:
            print(f"  {'':<{BAR_WIDTH}}         {human_bytes(committed)} committed da Windows "
                  f"(include memoria sfrattabile)")
        shared = g.get("shared_used_bytes")
        if shared:
            print(f"  {'':<{BAR_WIDTH}}         {human_bytes(shared)} su RAM di sistema "
                  f"— attenzione: rallenta molto")
        line = []
        if g.get("util_gpu_pct") is not None:
            line.append(f"utilizzo {g['util_gpu_pct']}%")
        if g.get("temperature_c") is not None:
            line.append(f"{g['temperature_c']} °C")
        if g.get("power_w"):
            line.append(f"{g['power_w']:.0f} W / {g.get('power_limit_w', 0):.0f} W")
        if g.get("clock_sm_mhz"):
            line.append(f"SM {g['clock_sm_mhz']} MHz")
        if line:
            print(f"  {' · '.join(line)}")
        if verbose:
            print(f"  BAR1 {human_bytes(g.get('bar1_used_bytes'))} / "
                  f"{human_bytes(g.get('bar1_total_bytes'))}")
            print(f"  {g.get('residency_note', '')}")

    balloon = status.get("balloon") or {}
    if balloon.get("allocated_bytes"):
        print(f"\n  Riserva attiva: {human_bytes(balloon['allocated_bytes'])}"
              f" (+{human_bytes(balloon.get('context_overhead_bytes', 0))} di contesto CUDA)")

    for w in status.get("warnings", []):
        print(f"  ! {w}")

    if verbose:
        h = status.get("health", {})
        print(f"\n  NVML={h.get('nvml_available')} PDH={h.get('pdh_available')} "
              f"NVAPI={h.get('nvapi_available')} campioni={h.get('samples')}")
        if h.get("last_error"):
            print(f"  ultimo errore: {h['last_error']}")


def print_processes(processes: list[dict], limit: int, show_actions: bool = False) -> None:
    if not processes:
        print("  nessun processo sta usando la VRAM")
        return
    print(f"\n  {'PID':>7}  {'DEDICATA':>10}  {'SU RAM':>9}  {'CONTENUTO':<44} FIDUCIA")
    print(f"  {'-' * 7}  {'-' * 10}  {'-' * 9}  {'-' * 44} {'-' * 9}")
    for p in processes[:limit]:
        content = p.get("content") or {}
        label = (content.get("label") or p.get("name") or "?")[:44]
        shared = p.get("shared_bytes", 0)
        flag = " !" if shared > 64 * 1024 * 1024 else "  "
        print(f"  {p['pid']:>7}  {human_bytes(p['dedicated_bytes']):>10}  "
              f"{human_bytes(shared):>9}{flag}{label:<44} {content.get('confidence', '?')}")
        detail = content.get("detail")
        if detail:
            print(f"  {'':>7}  {'':>10}  {'':>9}  → {detail[:88]}")
        if show_actions:
            for a in p.get("actions", []):
                if a.get("available"):
                    print(f"  {'':>7}  {'':>10}  {'':>9}    [{a['kind']}] {a['label']}")
    hidden = len(processes) - limit
    if hidden > 0:
        print(f"\n  ... e altri {hidden} processi (usare --limit per vederli)")


# --- comandi ---------------------------------------------------------------
def cmd_status(client: Client, ns: argparse.Namespace) -> int:
    status = client.get("/status") if client.remote else client.service.status()
    if ns.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0
    origin = "servizio" if client.remote else "istanza locale"
    print(f"\nVRAM EXPLORER — stato ({origin})")
    print_status(status, ns.verbose)
    print()
    return 0


def cmd_top(client: Client, ns: argparse.Namespace) -> int:
    if client.remote:
        status = client.get("/status")
        procs = client.get("/processes", include_system=not ns.no_system)["processes"]
    else:
        status = client.service.status()
        procs = client.service.processes(include_system=not ns.no_system)
    if ns.json:
        print(json.dumps({"status": status, "processes": procs}, indent=2,
                         ensure_ascii=False))
        return 0
    print("\nVRAM EXPLORER")
    print_status(status, ns.verbose)
    print_processes(procs, ns.limit, ns.actions)
    print()
    return 0


def cmd_watch(client: Client, ns: argparse.Namespace) -> int:
    """Aggiornamento continuo, come `top`."""
    try:
        while True:
            if client.remote:
                status = client.get("/status")
                procs = client.get("/processes", include_system=not ns.no_system)["processes"]
            else:
                status = client.service.status()
                procs = client.service.processes(include_system=not ns.no_system)
            print("\033[2J\033[H", end="")   # pulisce lo schermo
            print(f"VRAM EXPLORER — {time.strftime('%H:%M:%S')}"
                  f"   (Ctrl+C per uscire)")
            print_status(status, ns.verbose)
            print_processes(procs, ns.limit)
            time.sleep(max(0.5, ns.interval))
    except KeyboardInterrupt:
        print("\n")
        return 0


def cmd_unload(client: Client, ns: argparse.Namespace) -> int:
    payload = {"pid": ns.pid, "kind": ns.kind, "confirmed": ns.yes}
    if client.remote:
        result = client.post("/action", payload)
    else:
        entry = client.service.find_process(ns.pid)
        if entry is None:
            print(f"nessun processo {ns.pid} sta usando la GPU", file=sys.stderr)
            return 1
        result = client.service.unloader.execute(entry.info, ns.kind,
                                                 confirmed=ns.yes).to_dict()
    ok = result.get("ok")
    print(("✓ " if ok else "✗ ") + str(result.get("message", "")))
    if ok and result.get("freed_bytes"):
        print(f"  liberati {result['freed_human']}")
    elif ok:
        print("  nessuna memoria liberata secondo la misura immediata; "
              "alcuni backend impiegano qualche secondo")
    return 0 if ok else 1


def cmd_actions(client: Client, ns: argparse.Namespace) -> int:
    if client.remote:
        data = client.get(f"/process/{ns.pid}")
    else:
        entry = client.service.find_process(ns.pid)
        if entry is None:
            print(f"nessun processo {ns.pid} sta usando la GPU", file=sys.stderr)
            return 1
        data = entry.to_dict()
        data["actions"] = [o.to_dict()
                           for o in client.service.unloader.options_for(entry.info)]
    content = data.get("content") or {}
    print(f"\n  pid {ns.pid} — {content.get('label', data.get('name'))}")
    print(f"  {human_bytes(data.get('dedicated_bytes', 0))} dedicati, "
          f"{human_bytes(data.get('shared_bytes', 0))} su RAM\n")
    for a in data.get("actions", []):
        mark = "  " if a.get("available") else "✗ "
        warn = " [DISTRUTTIVA]" if a.get("destructive") else ""
        print(f"  {mark}{a['kind']:<20} {a['label']}{warn}")
        print(f"      {a['description']}")
        if not a.get("available") and a.get("unavailable_reason"):
            print(f"      non disponibile: {a['unavailable_reason']}")
    print()
    return 0


def cmd_reserve(client: Client, ns: argparse.Namespace) -> int:
    if ns.size is None:
        state = client.get("/reserve") if client.remote else client.service.balloon.state()
        print(json.dumps(state, indent=2, ensure_ascii=False) if ns.json
              else f"riserva attuale: {human_bytes(state.get('allocated_bytes', 0))}")
        return 0
    size = parse_size(ns.size)
    if client.remote:
        state = client.post("/reserve", {"size": size})
    else:
        state = client.service.balloon.set_reserve(size)
    print(f"riserva impostata: {human_bytes(state.get('allocated_bytes', 0))}")
    if state.get("context_overhead_bytes"):
        print(f"  il contesto CUDA del processo di riserva occupa in più "
              f"{human_bytes(state['context_overhead_bytes'])}")
    if state.get("error"):
        print(f"  ! {state['error']}")
    print("  nota: la riserva è efficace contro i client CUDA ma non impedisce "
          "al compositor di ottenere memoria")
    return 0


def cmd_policy(client: Client, ns: argparse.Namespace) -> int:
    if ns.set is None:
        state = (client.get("/policy", exe=ns.exe) if client.remote
                 else client.service.policy.get(ns.exe).to_dict())
        if ns.json:
            print(json.dumps(state, indent=2, ensure_ascii=False))
            return 0
        print(f"\n  ambito: {state['scope']}")
        print(f"  impostazione: {state['label']}")
        print(f"  {state['explanation']}")
        if state.get("is_default"):
            print("  (mai modificata: vale il comportamento predefinito del driver)")
        print()
        return 0

    value = 1 if ns.set in ("no-fallback", "fail", "1") else 0
    if client.remote:
        result = client.post("/policy", {"value": value, "exe": ns.exe})
    else:
        ok, message = client.service.policy.set(value, ns.exe)
        result = {"ok": ok, "message": message}
    print(("✓ " if result.get("ok") else "✗ ") + str(result.get("message")))
    return 0 if result.get("ok") else 1


def cmd_plan(client: Client, ns: argparse.Namespace) -> int:
    if client.remote:
        result = client.get("/plan", model=ns.model, context=ns.context,
                            kv_type=ns.kv_type)
    else:
        result = client.service.plan(ns.model, ns.context, kv_type=ns.kv_type)
    if ns.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if not result.get("ok"):
        print(f"✗ {result.get('error')}", file=sys.stderr)
        return 1

    model = result.get("model", {})
    print(f"\n  {model.get('name')}")
    print(f"  {human_bytes(model.get('size_bytes', 0))} · {model.get('quantization') or '?'}"
          f" · {model.get('n_layers') or '?'} layer\n")
    print(f"  {result['verdict']}\n")
    est = result.get("estimate") or {}
    if est:
        print(f"  pesi          {human_bytes(est['weights_bytes']):>10}")
        print(f"  cache KV      {human_bytes(est['kv_cache_bytes']):>10}"
              f"   (contesto {est.get('context_length', 0):,})".replace(",", " "))
        print(f"  overhead      {human_bytes(est['overhead_bytes']):>10}")
        print(f"  {'-' * 26}")
        print(f"  totale        {human_bytes(est['total_bytes']):>10}")
        if not est.get("is_estimate"):
            print(f"  {est.get('confidence_note')}")
        for w in est.get("warnings", []):
            print(f"\n  ! {w}")
    for step in result.get("plan_steps", []):
        print(f"\n  → {step}")
    for alt in result.get("alternatives", []):
        print(f"\n  ~ {alt}")
    print()
    return 0


def cmd_library(client: Client, ns: argparse.Namespace) -> int:
    if client.remote:
        data = client.get("/library", rescan=ns.rescan, query=ns.query or "",
                          limit=ns.limit)
    else:
        index = client.service.library_index(rescan=ns.rescan)
        models = index.find(ns.query) if ns.query else index.models
        data = {"count": len(models), "total_bytes": index.total_bytes(),
                "models": [m.to_dict() for m in models[:ns.limit]],
                "errors": index.errors}
    if ns.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    print(f"\n  {data['count']} modelli · {human_bytes(data['total_bytes'])} totali\n")
    for m in data["models"]:
        extra = []
        if m.get("quantization"):
            extra.append(m["quantization"])
        if m.get("n_layers"):
            extra.append(f"{m['n_layers']}L")
        if m.get("n_kv_heads"):
            extra.append(f"{m['n_kv_heads']} teste KV")
        print(f"  {human_bytes(m['size_bytes']):>9}  {m['fmt']:<12} {m['name'][:52]:<52}"
              f" {' · '.join(extra)}")
    for e in data.get("errors", [])[:5]:
        print(f"  ! {e}")
    print()
    return 0


def cmd_profiles(client: Client, ns: argparse.Namespace) -> int:
    if ns.apply or ns.preview:
        name = ns.apply or ns.preview
        if ns.apply and not ns.yes:
            # L'anteprima non e' un suggerimento: e' il passaggio obbligato prima
            # di riconfigurare la macchina.
            print("mostro prima l'anteprima; ripetere con --yes per applicare\n")
        payload = {"name": name, "confirmed": bool(ns.yes)}
        if ns.apply and ns.yes:
            result = (client.post("/profiles/apply", payload) if client.remote
                      else client.service.profiles.apply(name, client.service.snapshot(), True))
            print(f"\n  {result.get('summary', result.get('error'))}")
            for s in result.get("steps", []):
                mark = "✓" if s.get("ok") else "✗"
                print(f"  {mark} {s.get('kind')}: {s.get('message')}")
            print()
            return 0 if result.get("ok") else 1

        result = (client.post("/profiles/preview", payload) if client.remote
                  else client.service.profiles.preview(name, client.service.snapshot()))
        if not result.get("ok"):
            print(f"✗ {result.get('error')}", file=sys.stderr)
            return 1
        p = result["profile"]
        print(f"\n  {p['title']} — {p['description']}\n")
        for item in result["items"]:
            mark = "  " if item["applicable"] else "· "
            free = f"  (~{item['estimated_free_human']})" if item.get("estimated_free_human") else ""
            print(f"  {mark}{item['title']}{free}")
            print(f"      {item['detail']}")
            if item.get("warning"):
                print(f"      ! {item['warning']}")
        print(f"\n  totale liberabile stimato: {result['estimated_total_free_human']}")
        print(f"  passi applicabili: {result['applicable_steps']}/{result['total_steps']}\n")
        return 0

    profiles = (client.get("/profiles")["profiles"] if client.remote
                else [p.to_dict() for p in client.service.profiles.list()])
    print()
    for p in profiles:
        tag = " (predefinito)" if p.get("builtin") else ""
        print(f"  {p['name']:<20} {p['title']}{tag}")
        print(f"  {'':<20} {p['description'][:88]}")
    print()
    return 0


def cmd_incidents(client: Client, ns: argparse.Namespace) -> int:
    if client.remote:
        rows = client.get("/incidents", limit=ns.limit)["incidents"]
    else:
        rows = client.service.store.incidents(limit=ns.limit)
    if ns.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("\n  nessun incidente registrato\n")
        return 0
    print()
    for r in rows:
        when = time.strftime("%d/%m %H:%M:%S", time.localtime(r["ts"]))
        who = r.get("process_name") or "sistema"
        print(f"  {when}  [{r['kind']}]  {who} (pid {r.get('pid')})")
        print(f"      {r.get('detail', '')[:110]}")
    print()
    return 0


def cmd_probe(client: Client, ns: argparse.Namespace) -> int:
    from ..probe.attach import check_eligibility
    if ns.attach:
        eligible, reason = check_eligibility(ns.pid)
        print(f"\n  idoneita': {'sì' if eligible else 'no'} — {reason}")
        if not eligible:
            return 1
        if not ns.yes:
            print("\n  L'iniezione a caldo può destabilizzare il processo target.")
            print("  Ripetere con --yes per procedere, oppure riavviare il processo")
            print("  con VRAMEXP_AGENT=1, che è la via sicura.\n")
            return 1
        from ..probe.attach import attach
        result = attach(ns.pid, confirmed=True, enabled=client.cfg.enable_hot_attach)
        print(("✓ " if result.ok else "✗ ") + result.message)
        return 0 if result.ok else 1

    from ..probe.client import AgentClient
    agent = AgentClient()
    data = agent.call(ns.pid, "stats")
    if not data or not data.get("ok"):
        print(f"\n  agent non attivo nel processo {ns.pid}.")
        print("  Avviare il processo con VRAMEXP_AGENT=1 oppure usare --attach.\n")
        return 1
    print(f"\n  pid {ns.pid}")
    print(f"  allocato        {human_bytes(data['allocated_bytes']):>10}")
    print(f"  riservato       {human_bytes(data['reserved_bytes']):>10}")
    print(f"  frammentazione  {human_bytes(data['fragmentation_bytes']):>10}"
          f"   ({pct(data['fragmentation_bytes'], data['reserved_bytes']):.0f}% del riservato)")
    print(f"  picco allocato  {human_bytes(data['max_allocated_bytes']):>10}")
    print(f"  retry alloc     {data['num_alloc_retries']:>10}"
          "   (>0 indica pressione sulla memoria)")
    print(f"  OOM             {data['num_ooms']:>10}")

    snap = agent.call(ns.pid, "snapshot", limit=ns.limit)
    if snap and snap.get("ok"):
        print(f"\n  {snap['block_count']} blocchi attivi, "
              f"{human_bytes(snap['total_bytes'])} in totale\n")
        for b in snap["blocks"][:ns.limit]:
            frames = b.get("frames") or []
            origin = frames[0] if frames else "origine ignota (attivare record)"
            print(f"  {human_bytes(b['size']):>10}  ← {origin[-92:]}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vramexp", description="Esplorazione e gestione della VRAM")
    p.add_argument("--json", action="store_true", help="output in JSON")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--local", action="store_true",
                   help="non usare il servizio, lavora in proprio")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="stato della GPU")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("top", help="processi ordinati per VRAM occupata")
    s.add_argument("--limit", type=int, default=15)
    s.add_argument("--no-system", action="store_true", help="nascondi i processi di Windows")
    s.add_argument("--actions", action="store_true", help="mostra le azioni disponibili")
    s.set_defaults(func=cmd_top)

    s = sub.add_parser("watch", help="aggiornamento continuo")
    s.add_argument("--interval", type=float, default=2.0)
    s.add_argument("--limit", type=int, default=12)
    s.add_argument("--no-system", action="store_true")
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("actions", help="azioni disponibili per un processo")
    s.add_argument("pid", type=int)
    s.set_defaults(func=cmd_actions)

    s = sub.add_parser("unload", help="scarica contenuto dalla VRAM")
    s.add_argument("pid", type=int)
    s.add_argument("--kind", default="backend_api",
                   choices=["backend_api", "agent_empty_cache", "trim", "suspend", "kill"])
    s.add_argument("--yes", action="store_true", help="conferma le azioni distruttive")
    s.set_defaults(func=cmd_unload)

    s = sub.add_parser("reserve", help="imposta o legge la riserva di VRAM")
    s.add_argument("size", nargs="?", default=None, help='es. 8G, 512M, 0 per rilasciare')
    s.set_defaults(func=cmd_reserve)

    s = sub.add_parser("policy", help="policy di fallback sulla RAM di sistema")
    s.add_argument("--set", choices=["allow", "no-fallback"], default=None)
    s.add_argument("--exe", default=None, help="applica a un solo eseguibile")
    s.set_defaults(func=cmd_policy)

    s = sub.add_parser("plan", help="verifica se un modello entra in VRAM")
    s.add_argument("model", help="nome o frammento del nome")
    s.add_argument("--context", type=int, default=None)
    s.add_argument("--kv-type", default="f16", choices=["f16", "q8_0", "q5_1", "q4_0"])
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("library", help="modelli presenti su disco")
    s.add_argument("query", nargs="?", default=None)
    s.add_argument("--rescan", action="store_true")
    s.add_argument("--limit", type=int, default=30)
    s.set_defaults(func=cmd_library)

    s = sub.add_parser("profiles", help="profili operativi")
    s.add_argument("--preview", default=None, metavar="NOME")
    s.add_argument("--apply", default=None, metavar="NOME")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_profiles)

    s = sub.add_parser("incidents", help="eventi registrati (OOM, spillover)")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_incidents)

    s = sub.add_parser("probe", help="introspezione di un processo PyTorch")
    s.add_argument("pid", type=int)
    s.add_argument("--limit", type=int, default=12)
    s.add_argument("--attach", action="store_true", help="inietta l'agent a caldo")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_probe)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    cfg = Config.load()
    client = Client(cfg, force_local=ns.local)
    try:
        return ns.func(client, ns)
    except KeyboardInterrupt:
        return 130
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text[:200]
        print(f"errore dal servizio: {detail or e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        if ns.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
