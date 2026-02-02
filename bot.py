from __future__ import annotations

"""
bot.py — Oráculo Lotofácil (Preset Mestre) — padrão "Lotomania"

Melhorias:
- Determinismo real: mesma base (último resultado) => mesmo lote, sempre.
- Persistência: data/last_batch.json (sobrevive restart).
- Watcher: detecta novo resultado no data/history.csv, alerta e (opcional) treina.
- Auditoria: data/audit_log.jsonl (1 evento por linha).
- Comandos: /status, /resultado (admin), /regerar.

Compatibilidade JobQueue:
- Se python-telegram-bot estiver com extra [job-queue], usa JobQueue.
- Se NÃO estiver, roda watcher via asyncio (fallback), sem quebrar deploy.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from engine_oraculo import OraculoEngine, EngineConfig, shape_ok_mestre, paridade, max_seq
from learning import LearningCore, LearnConfig

# --------------------------- Logging ---------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("lotofacil_oraculo")

# --------------------------- Env / Config ---------------------------

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
TIMEZONE = os.environ.get("TZ", "America/Sao_Paulo")

HISTORY_PATH = os.environ.get("HISTORY_PATH", "data/history.csv")
LEARN_STATE_PATH = os.environ.get("LEARN_STATE_PATH", "data/learn_state.json")

LAST_BATCH_PATH = os.environ.get("LAST_BATCH_PATH", "data/last_batch.json")
LAST_SEEN_PATH = os.environ.get("LAST_SEEN_PATH", "data/last_seen.json")
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "data/audit_log.jsonl")

AUTO_ALERT_ON_NEW_RESULT = os.getenv("AUTO_ALERT_ON_NEW_RESULT", "1") == "1"
AUTO_TRAIN_ON_NEW_RESULT = os.getenv("AUTO_TRAIN_ON_NEW_RESULT", "0") == "1"
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID", "").strip()
RESULT_CHECK_INTERVAL_SEC = int(os.getenv("RESULT_CHECK_INTERVAL_SEC", "300"))

_admin_ids_env = os.getenv("ADMIN_IDS", "").strip()
if _admin_ids_env:
    try:
        ADMIN_IDS: Set[int] = {int(x.strip()) for x in _admin_ids_env.split(",") if x.strip()}
    except Exception:
        ADMIN_IDS = set()
else:
    ADMIN_IDS: Set[int] = {5344714174}

WHITELIST_PATH = os.environ.get("WHITELIST_PATH", "whitelist.txt")

COOLDOWN_SECONDS = float(os.getenv("COOLDOWN_SECONDS", "8.0"))

SEED_SALT = os.getenv("SEED_SALT", "mestre_lotofacil_salt_v1")
ALGO_VERSION = os.getenv("ALGO_VERSION", "mestre_v2")

ENGINE_OVERLAP_MAX = int(os.getenv("ENGINE_OVERLAP_MAX", "11"))
ENGINE_TARGET_QTD = int(os.getenv("ENGINE_TARGET_QTD", "10"))

# --------------------------- Runtime State ---------------------------

LAST_APOSTAS: List[List[int]] = []
LAST_BASE: List[int] = []
_last_call_per_user: Dict[Tuple[int, str], float] = {}

# --------------------------- Helpers: filesystem ---------------------------

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _now_iso() -> str:
    try:
        tz = ZoneInfo(TIMEZONE)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).replace(microsecond=0).isoformat()

def _append_audit(event: str, payload: dict) -> None:
    try:
        _ensure_parent_dir(AUDIT_LOG_PATH)
        rec = {"ts": _now_iso(), "event": event, **payload}
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Falha ao escrever auditoria: %s", e)

def _save_last_batch(apostas: List[List[int]], base: List[int], seed: int) -> None:
    global LAST_APOSTAS, LAST_BASE
    LAST_APOSTAS = [sorted([int(x) for x in a]) for a in apostas]
    LAST_BASE = sorted([int(x) for x in base])
    data = {
        "ts": _now_iso(),
        "seed": int(seed),
        "base": LAST_BASE,
        "apostas": LAST_APOSTAS,
        "algo_version": ALGO_VERSION,
    }
    _ensure_parent_dir(LAST_BATCH_PATH)
    with open(LAST_BATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_last_batch() -> bool:
    global LAST_APOSTAS, LAST_BASE
    try:
        if not os.path.exists(LAST_BATCH_PATH):
            return False
        with open(LAST_BATCH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data.get("base", [])
        apostas = data.get("apostas", [])
        if not isinstance(base, list) or not isinstance(apostas, list):
            return False

        base = sorted([int(x) for x in base if 1 <= int(x) <= 25])
        if len(base) != 15 or len(set(base)) != 15:
            return False

        cleaned: List[List[int]] = []
        for a in apostas:
            if not isinstance(a, list):
                continue
            aa = sorted([int(x) for x in a if 1 <= int(x) <= 25])
            if len(aa) == 15 and len(set(aa)) == 15:
                cleaned.append(aa)

        if not cleaned:
            return False

        LAST_BASE = base
        LAST_APOSTAS = cleaned
        return True
    except Exception as e:
        logger.warning("Falha ao carregar last_batch: %s", e)
        return False

def _read_last_seen_sig() -> str:
    try:
        if not os.path.exists(LAST_SEEN_PATH):
            return ""
        with open(LAST_SEEN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("sig", "")).strip()
    except Exception:
        return ""

def _write_last_seen_sig(sig: str) -> None:
    _ensure_parent_dir(LAST_SEEN_PATH)
    with open(LAST_SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump({"ts": _now_iso(), "sig": sig}, f, ensure_ascii=False, indent=2)

# --------------------------- Helpers: security / access ---------------------------

def _hit_cooldown(user_id: int, comando: str, cooldown: float = COOLDOWN_SECONDS) -> bool:
    import time
    key = (user_id, comando)
    now = time.time()
    last = _last_call_per_user.get(key, 0.0)
    if now - last < cooldown:
        return True
    _last_call_per_user[key] = now
    return False

def _is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

def _load_whitelist() -> Set[int]:
    ids: Set[int] = set()
    try:
        if not os.path.exists(WHITELIST_PATH):
            return ids
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(int(line))
                except Exception:
                    continue
    except Exception:
        pass
    return ids

def _usuario_autorizado(user_id: int) -> bool:
    if _is_admin(user_id):
        return True
    wl = _load_whitelist()
    return user_id in wl

# --------------------------- Helpers: history parsing ---------------------------

def _parse_dezenas_line(line: str) -> Optional[List[int]]:
    if ";" in line:
        line = line.split(";", 1)[0]
    line = line.strip()
    if not line:
        return None

    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) < 15:
        return None

    try:
        nums = [int(p) for p in parts[:15]]
    except Exception:
        return None

    nums = [n for n in nums if 1 <= n <= 25]
    if len(nums) != 15 or len(set(nums)) != 15:
        return None

    return sorted(nums)

def carregar_historico(path: str) -> List[List[int]]:
    if not os.path.exists(path):
        return []
    out: List[List[int]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                dezenas = _parse_dezenas_line(raw)
                if dezenas:
                    out.append(dezenas)
    except Exception as e:
        logger.warning("Falha ao ler history: %s", e)
        return []
    return out

def ultimo_resultado(historico: List[List[int]]) -> List[int]:
    if not historico:
        return []
    return list(sorted(historico[0]))

def _sig_from_dezenas(dezenas: List[int]) -> str:
    s = ",".join(f"{d:02d}" for d in sorted(dezenas))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# --------------------------- Deterministic seed ---------------------------

def deterministic_seed(base: List[int]) -> int:
    base_s = ",".join(f"{d:02d}" for d in sorted(base))
    payload = f"{ALGO_VERSION}|{SEED_SALT}|{base_s}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF

# --------------------------- Presentation helpers ---------------------------

def _format_aposta(a: List[int]) -> str:
    return " ".join(f"{x:02d}" for x in sorted(a))

def _placar(aposta: List[int], oficial: List[int]) -> int:
    s = set(oficial)
    return sum(1 for x in aposta if x in s)

# --------------------------- Commands ---------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0

    # Se quiser permitir /start até para não liberados, remova esta checagem.
    if not _usuario_autorizado(uid):
        return await update.message.reply_text(
            "⛔ Acesso não autorizado.\n"
            "Use /meuid e envie seu ID ao administrador para liberação."
        )

    # Menu para usuário comum
    if not _is_admin(uid):
        return await update.message.reply_text(
            "✅ Oráculo Lotofácil — Preset Mestre\n\n"
            "Comandos disponíveis:\n"
            "/gerar — gera 10 apostas (determinístico)\n"
            "/meuid — mostra seu ID\n"
        )

    # Menu para ADMIN (somente admin vê isto)
    return await update.message.reply_text(
        "✅ Oráculo Lotofácil — Preset Mestre (ADMIN)\n\n"
        "Comandos:\n"
        "/gerar — gera 10 apostas (determinístico)\n"
        "/regerar — regenera o mesmo lote da base atual\n"
        "/status — status do sistema\n"
        "/meuid — mostra seu ID\n\n"
        "Admin:\n"
        "/confirmar <15 dezenas>\n"
        "/resultado <15 dezenas>\n"
    )


async def meuid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0
    return await update.message.reply_text(f"Seu ID é: {uid}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0

    if not _usuario_autorizado(uid):
        return await update.message.reply_text("⛔ Acesso não autorizado.")

    learn = LearningCore(LearnConfig(state_path=LEARN_STATE_PATH))
    hist = carregar_historico(HISTORY_PATH)
    last = ultimo_resultado(hist)
    last_seen = _read_last_seen_sig()

    msg = []
    msg.append("📌 Status — Oráculo Lotofácil (Mestre)")
    msg.append(f"Último resultado: {_format_aposta(last) if last else '(histórico vazio)'}")
    msg.append(f"Alpha: {learn.get_alpha():.3f}")
    msg.append(f"Auto-alert: {AUTO_ALERT_ON_NEW_RESULT} | Auto-train: {AUTO_TRAIN_ON_NEW_RESULT} | Intervalo: {RESULT_CHECK_INTERVAL_SEC}s")
    msg.append(f"Last-seen set: {'SIM' if last_seen else 'NÃO'}")
    msg.append(f"Último lote em memória: {len(LAST_APOSTAS) if LAST_APOSTAS else 0}")

    if _is_admin(uid):
        bias = learn.get_bias_num() or {}
        if bias:
            plus = sorted(bias.items(), key=lambda kv: kv[1], reverse=True)[:5]
            minus = sorted(bias.items(), key=lambda kv: kv[1])[:5]
            msg.append("Top bias +: " + ", ".join([f"{int(k):02d}:{v:+.2f}" for k, v in plus]))
            msg.append("Top bias -: " + ", ".join([f"{int(k):02d}:{v:+.2f}" for k, v in minus]))

    _append_audit("status", {"user_id": uid})
    return await update.message.reply_text("\n".join(msg))

async def gerar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0

    if not _usuario_autorizado(uid):
        return await update.message.reply_text("⛔ Acesso não autorizado.")

    if _hit_cooldown(uid, "gerar", cooldown=COOLDOWN_SECONDS):
        return await update.message.reply_text("⏳ Aguarde alguns segundos antes de gerar novamente.")

    hist = carregar_historico(HISTORY_PATH)
    base = ultimo_resultado(hist)
    if not base:
        return await update.message.reply_text("⚠️ Histórico vazio ou inválido. Verifique HISTORY_PATH.")

    learn = LearningCore(LearnConfig(state_path=LEARN_STATE_PATH))
    cfg = EngineConfig(overlap_max=ENGINE_OVERLAP_MAX, target_qtd=ENGINE_TARGET_QTD)
    engine = OraculoEngine(config=cfg, alpha=learn.get_alpha(), bias_num=learn.get_bias_num())

    seed = deterministic_seed(base)

    try:
        apostas = engine.gerar_lote(ultimo_resultado=base, qtd=ENGINE_TARGET_QTD, seed=seed)
    except TypeError:
        apostas = engine.gerar_lote(base, qtd=ENGINE_TARGET_QTD, seed=seed)
    except Exception as e:
        logger.error("Erro ao gerar lote: %s", e, exc_info=True)
        return await update.message.reply_text(f"❌ Erro ao gerar lote: {e}")

    if not apostas or len(apostas) != ENGINE_TARGET_QTD:
        return await update.message.reply_text("⚠️ Não foi possível gerar o lote completo dentro das regras.")

    _save_last_batch(apostas, base, seed)
    _append_audit("gerar", {"user_id": uid, "seed": seed, "base": base, "qtd": len(apostas)})

    try:
        learn.registrar_lote_gerado(base=base, apostas=apostas, tag="gerar")
    except TypeError:
        try:
            learn.registrar_lote_gerado(oficial_base=base, apostas=apostas, tag="gerar")
        except Exception:
            pass
    except Exception:
        pass

    linhas = []
    linhas.append("🎰 SUAS APOSTAS INTELIGENTES — Preset Mestre 🎰")
    linhas.append(f"Base: {_format_aposta(base)}")
    linhas.append(f"Seed determinístico: {seed}")
    linhas.append("")

    for i, a in enumerate(apostas, 1):
        pares, imp = paridade(a)
        seq = max_seq(a)
        r = len(set(a) & set(base))
        hit = _placar(a, base)
        linhas.append(f"Aposta {i}: {_format_aposta(a)}")
        linhas.append(
            f"🔢 Pares: {pares} | Ímpares: {imp} | SeqMax: {seq} | {r}R | "
            f"{hit} acertos (vs. último) | {'✅ OK' if shape_ok_mestre(a) else '🛠️ REVER'}"
        )
        linhas.append("")

    return await update.message.reply_text("\n".join(linhas))

async def regerar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0

    if not _usuario_autorizado(uid):
        return await update.message.reply_text("⛔ Acesso não autorizado.")

    if _hit_cooldown(uid, "regerar", cooldown=COOLDOWN_SECONDS):
        return await update.message.reply_text("⏳ Aguarde alguns segundos antes de usar /regerar novamente.")

    hist = carregar_historico(HISTORY_PATH)
    base = ultimo_resultado(hist)
    if not base:
        return await update.message.reply_text("⚠️ Histórico vazio ou inválido.")

    learn = LearningCore(LearnConfig(state_path=LEARN_STATE_PATH))
    cfg = EngineConfig(overlap_max=ENGINE_OVERLAP_MAX, target_qtd=ENGINE_TARGET_QTD)
    engine = OraculoEngine(config=cfg, alpha=learn.get_alpha(), bias_num=learn.get_bias_num())

    seed = deterministic_seed(base)

    try:
        apostas = engine.gerar_lote(ultimo_resultado=base, qtd=ENGINE_TARGET_QTD, seed=seed)
    except TypeError:
        apostas = engine.gerar_lote(base, qtd=ENGINE_TARGET_QTD, seed=seed)

    _save_last_batch(apostas, base, seed)
    _append_audit("regerar", {"user_id": uid, "seed": seed, "base": base, "qtd": len(apostas)})

    return await update.message.reply_text(
        "✅ Regerado o mesmo lote da base atual.\n"
        f"Base: {_format_aposta(base)}\n"
        f"Seed: {seed}"
    )

async def confirmar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0

    if not _is_admin(uid):
        return await update.message.reply_text("⛔ Este comando é restrito ao administrador.")

    if _hit_cooldown(uid, "confirmar", cooldown=4.0):
        return await update.message.reply_text("⏳ Aguarde alguns segundos antes de usar /confirmar novamente.")

    partes = (update.message.text or "").strip().split()
    dezenas_raw = partes[1:]

    try:
        dezenas = [int(x) for x in dezenas_raw]
    except Exception:
        return await update.message.reply_text("Use: /confirmar <15 dezenas entre 1..25>")

    if len(dezenas) != 15 or any(d < 1 or d > 25 for d in dezenas) or len(set(dezenas)) != 15:
        return await update.message.reply_text("❗ Envie exatamente 15 dezenas únicas entre 1–25.")

    oficial = sorted(dezenas)

    if not LAST_APOSTAS:
        _load_last_batch()
    if not LAST_APOSTAS:
        return await update.message.reply_text("⚠️ Ainda não há lote disponível. Use /gerar antes.")

    learn = LearningCore(LearnConfig(state_path=LEARN_STATE_PATH))
    alpha_before = learn.get_alpha()

    try:
        rel = learn.aprender_com_lote(oficial=oficial, apostas=LAST_APOSTAS, tag="confirmar")
    except Exception as e:
        logger.error("Erro no aprendizado /confirmar: %s", e, exc_info=True)
        return await update.message.reply_text(f"Erro no aprendizado: {e}")

    _append_audit("confirmar", {
        "user_id": uid,
        "oficial": oficial,
        "melhor": rel.get("melhor"),
        "media": rel.get("media"),
        "topk": rel.get("topk"),
        "alpha_before": alpha_before,
        "alpha_after": rel.get("alpha"),
    })

    melhor = rel.get("melhor", 0)
    media = rel.get("media", 0.0)
    topk = rel.get("topk", 0.0)
    alpha_after = rel.get("alpha", alpha_before)
    placares = rel.get("placares", [])

    linhas = [
        "✅ Confirmado e aprendido.",
        f"Oficial: {_format_aposta(oficial)}",
        f"Melhor: {melhor} | Média: {media:.2f} | TopK: {topk:.2f}",
        f"Alpha: {alpha_before:.3f} → {alpha_after:.3f}",
        "Placares: " + ", ".join(str(x) for x in placares),
    ]
    return await update.message.reply_text("\n".join(linhas))

async def resultado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0

    if not _is_admin(uid):
        return await update.message.reply_text("⛔ Este comando é restrito ao administrador.")

    if _hit_cooldown(uid, "resultado", cooldown=4.0):
        return await update.message.reply_text("⏳ Aguarde alguns segundos antes de usar /resultado novamente.")

    partes = (update.message.text or "").strip().split()
    dezenas_raw = partes[1:]

    try:
        dezenas = [int(x) for x in dezenas_raw]
    except Exception:
        return await update.message.reply_text("Use: /resultado <15 dezenas entre 1..25>")

    if len(dezenas) != 15 or any(d < 1 or d > 25 for d in dezenas) or len(set(dezenas)) != 15:
        return await update.message.reply_text("❗ Envie exatamente 15 dezenas únicas entre 1–25.")

    oficial = sorted(dezenas)

    _ensure_parent_dir(HISTORY_PATH)
    old = ""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            old = f.read()

    line = ",".join(str(x) for x in oficial) + "\n"
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        f.write(line + old)

    sig = _sig_from_dezenas(oficial)
    _write_last_seen_sig(sig)

    _append_audit("resultado_registrado", {"user_id": uid, "oficial": oficial})

    if AUTO_ALERT_ON_NEW_RESULT and ALERT_CHAT_ID:
        try:
            await context.bot.send_message(
                chat_id=int(ALERT_CHAT_ID),
                text=f"📣 Resultado registrado no history.csv:\n{_format_aposta(oficial)}",
            )
        except Exception as e:
            logger.warning("Falha ao enviar alerta: %s", e)

    trained = False
    if AUTO_TRAIN_ON_NEW_RESULT:
        if not LAST_APOSTAS:
            _load_last_batch()
        if LAST_APOSTAS:
            learn = LearningCore(LearnConfig(state_path=LEARN_STATE_PATH))
            try:
                rel = learn.aprender_com_lote(oficial=oficial, apostas=LAST_APOSTAS, tag="auto_train(resultado)")
                trained = True
                _append_audit("auto_train", {
                    "oficial": oficial,
                    "melhor": rel.get("melhor"),
                    "media": rel.get("media"),
                    "alpha": rel.get("alpha"),
                })
            except Exception as e:
                logger.error("Auto-train falhou: %s", e, exc_info=True)

    return await update.message.reply_text(
        "✅ Resultado registrado no history.csv.\n"
        f"Oficial: {_format_aposta(oficial)}\n"
        f"Auto-train: {'✅' if trained else '⏸️ (desligado ou sem lote)'}"
    )

# --------------------------- Watcher core ---------------------------

async def _watch_new_result_app(app: Application) -> None:
    """
    Núcleo do watcher: roda 1 vez (sem depender de JobQueue).
    """
    hist = carregar_historico(HISTORY_PATH)
    if not hist:
        return

    last = ultimo_resultado(hist)
    sig = _sig_from_dezenas(last)
    last_seen = _read_last_seen_sig()

    if not last_seen:
        _write_last_seen_sig(sig)
        logger.info("Watcher first-run: marcando last_seen (sem alertar).")
        return

    if sig == last_seen:
        return

    _write_last_seen_sig(sig)
    _append_audit("novo_resultado_detectado", {"oficial": last})

    if AUTO_ALERT_ON_NEW_RESULT and ALERT_CHAT_ID:
        try:
            await app.bot.send_message(
                chat_id=int(ALERT_CHAT_ID),
                text=f"📣 Novo resultado detectado no history.csv:\n{_format_aposta(last)}",
            )
        except Exception as e:
            logger.warning("Falha ao enviar alerta: %s", e)

    if AUTO_TRAIN_ON_NEW_RESULT:
        if not LAST_APOSTAS:
            _load_last_batch()
        if LAST_APOSTAS:
            learn = LearningCore(LearnConfig(state_path=LEARN_STATE_PATH))
            try:
                rel = learn.aprender_com_lote(oficial=last, apostas=LAST_APOSTAS, tag="auto_train(watcher)")
                _append_audit("auto_train", {
                    "oficial": last,
                    "melhor": rel.get("melhor"),
                    "media": rel.get("media"),
                    "alpha": rel.get("alpha"),
                })
            except Exception as e:
                logger.error("Auto-train watcher falhou: %s", e, exc_info=True)

async def _watch_loop(app: Application) -> None:
    """
    Fallback quando não há JobQueue: loop asyncio.
    """
    await asyncio.sleep(10)
    while True:
        try:
            await _watch_new_result_app(app)
        except Exception as e:
            logger.error("Watcher(loop) erro: %s", e, exc_info=True)
        await asyncio.sleep(max(5, int(RESULT_CHECK_INTERVAL_SEC)))

async def _post_init(application: Application) -> None:
    """
    Roda ao iniciar o app:
    - Tenta agendar via JobQueue se existir
    - Senão cria task asyncio (fallback)
    """
    if RESULT_CHECK_INTERVAL_SEC <= 0:
        return

    jq = getattr(application, "job_queue", None)
    if jq is not None:
        try:
            jq.run_repeating(lambda ctx: _watch_new_result_app(application), interval=RESULT_CHECK_INTERVAL_SEC, first=10)
            logger.info("Watcher agendado via JobQueue.")
            return
        except Exception as e:
            logger.warning("Falha ao agendar via JobQueue, usando fallback asyncio: %s", e)

    application.create_task(_watch_loop(application))
    logger.info("Watcher agendado via asyncio fallback (sem JobQueue).")

# --------------------------- Generic handler ---------------------------

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return await update.message.reply_text("Comando não reconhecido. Use /start para ver os comandos.")

# --------------------------- App bootstrap ---------------------------

def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN não configurado.")

    app = ApplicationBuilder().token(BOT_TOKEN).post_init(_post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("meuid", meuid))
    app.add_handler(CommandHandler("gerar", gerar))
    app.add_handler(CommandHandler("regerar", regerar))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("confirmar", confirmar))
    app.add_handler(CommandHandler("resultado", resultado))

    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    return app

def main() -> None:
    _load_last_batch()

    app = build_app()
    logger.info(
        "Bot iniciado. Auto-alert=%s Auto-train=%s interval=%ss",
        AUTO_ALERT_ON_NEW_RESULT, AUTO_TRAIN_ON_NEW_RESULT, RESULT_CHECK_INTERVAL_SEC
    )
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
