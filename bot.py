from __future__ import annotations
import logging
import os
import json
import json
from typing import List, Dict, Set
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from engine_oraculo_v2 import OraculoEngine, EngineConfig, shape_ok_mestre, paridade, max_seq, deterministic_seed
from learning import LearningCore, LearnConfig, _hits

# --------------------------- Configuração ---------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("lotofacil_oraculo")

BOT_TOKEN = os.environ.get("BOT_TOKEN", "COLOQUE_SEU_TOKEN_AQUI")
TIMEZONE = os.environ.get("TZ", "America/Sao_Paulo")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "data/history.csv")

# Caminho do arquivo de estado de aprendizado (deve apontar para um volume no Railway)
LEARN_STATE_PATH = os.environ.get("LEARN_STATE_PATH", "data/learn_state.json")

# Persistência do último lote (sobrevive restart)
LAST_BATCH_PATH = os.environ.get("LAST_BATCH_PATH", "data/last_batch.json")
# Persistência de "último resultado visto" para auto-alert/auto-train
LAST_SEEN_PATH = os.environ.get("LAST_SEEN_PATH", "data/last_seen.json")
# Auditoria (1 evento por linha)
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "data/audit_log.jsonl")

# Automação estilo Lotomania
AUTO_ALERT_ON_NEW_RESULT = os.environ.get("AUTO_ALERT_ON_NEW_RESULT", "1") == "1"
AUTO_TRAIN_ON_NEW_RESULT = os.environ.get("AUTO_TRAIN_ON_NEW_RESULT", "0") == "1"
ALERT_CHAT_ID = os.environ.get("ALERT_CHAT_ID", "").strip()
RESULT_CHECK_INTERVAL_SEC = float(os.environ.get("RESULT_CHECK_INTERVAL_SEC", "300"))

# Admin fixo (não depende de variável de ambiente)
# Somente este ID terá acesso aos comandos administrativos
ADMIN_IDS: Set[int] = {5344714174}

# Caminho da whitelist de clientes pagantes (1 ID por linha)
WHITELIST_PATH = os.environ.get("WHITELIST_PATH", "whitelist.txt")


def carregar_whitelist(path: str) -> Set[int]:
    """
    Carrega whitelist.txt:
    - 1 ID numérico por linha
    - Ignora linhas vazias ou não numéricas
    """
    ids: Set[int] = set()
    if not os.path.exists(path):
        return ids
    try:
        with open(path, "r", encoding="utf-8") as f:
            for linha in f:
                s = linha.strip()
                if s.isdigit():
                    ids.add(int(s))
    except Exception as e:
        logger.error(f"Erro ao carregar whitelist de {path}: {e}", exc_info=True)
    return ids


# Conjunto de usuários liberados (clientes pagantes)
WHITELIST_IDS: Set[int] = carregar_whitelist(WHITELIST_PATH)

# ------------------------ Estado em memória (/gerar -> /confirmar) ------------------------

# Lote mais recente gerado pelo /gerar
LAST_APOSTAS: List[List[int]] = []
# Base (último resultado do history) usada para gerar esse lote
LAST_BASE: List[int] = []


# ------------------------ Persistência / Auditoria ------------------------

def _ensure_parent_dir(path: str) -> None:
    try:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def _load_last_batch() -> dict:
    try:
        if os.path.exists(LAST_BATCH_PATH):
            with open(LAST_BATCH_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        return {}
    return {}


def _save_last_batch(base: List[int], apostas: List[List[int]]) -> None:
    try:
        _ensure_parent_dir(LAST_BATCH_PATH)
        payload = {
            "ts": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
            "base": list(base),
            "apostas": [list(a) for a in apostas],
            "algo": "mestre_v2",
        }
        with open(LAST_BATCH_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _append_audit(event: dict) -> None:
    try:
        _ensure_parent_dir(AUDIT_LOG_PATH)
        event = dict(event)
        event.setdefault("ts", datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"))
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_latest_key_from_history_csv(path: str) -> tuple[str, List[int]]:
    """
    Retorna (chave_ultimo, dezenas_ultimo) a partir do history.csv.

    Como seu history.csv não tem 'concurso/data', usamos:
      key = "dezenas:02-03-..."
    e consideramos o ÚLTIMO concurso como a PRIMEIRA linha de dados (mais recente primeiro).

    Também ignora tudo depois de ';' (proteção caso alguém cole texto no final da linha).
    """
    if not os.path.exists(path):
        return ("", [])

    def parse_line(line: str) -> List[int]:
        line = (line or "").split(";", 1)[0].strip()
        if not line:
            return []
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 15:
            return []
        try:
            nums = [int(x) for x in parts[-15:]]
        except Exception:
            return []
        nums = [n for n in nums if 1 <= n <= 25]
        if len(nums) != 15:
            return []
        nums = sorted(set(nums))
        return nums if len(nums) == 15 else []

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    if not raw_lines:
        return ("", [])

    start_idx = 0
    if len(parse_line(raw_lines[0])) != 15:
        start_idx = 1
    if start_idx >= len(raw_lines):
        return ("", [])

    dezenas = parse_line(raw_lines[start_idx])
    if len(dezenas) != 15:
        return ("", [])

    key = "dezenas:" + "-".join(f"{d:02d}" for d in dezenas)
    return (key, dezenas)


def _read_last_seen_key() -> str:
    try:
        if os.path.exists(LAST_SEEN_PATH):
            with open(LAST_SEEN_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            return str(data.get("last_seen_key", "")).strip()
    except Exception:
        return ""
    return ""


def _write_last_seen_key(key: str) -> None:
    try:
        _ensure_parent_dir(LAST_SEEN_PATH)
        payload = {"last_seen_key": key, "ts": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")}
        with open(LAST_SEEN_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


async def _alert_and_train_if_new_result(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job periódico (estilo Lotomania): se detectar resultado novo no history.csv, alerta e treina."""
    key, dezenas = _get_latest_key_from_history_csv(HISTORY_PATH)
    if not key or len(dezenas) != 15:
        return

    last_seen = _read_last_seen_key()

    # primeira execução: só marca (não alerta)
    if not last_seen:
        _write_last_seen_key(key)
        _append_audit({"event": "bootstrap_last_seen", "key": key, "dezenas": dezenas})
        return

    if key == last_seen:
        return

    # mudou => novo resultado
    _write_last_seen_key(key)

    # alerta
    if AUTO_ALERT_ON_NEW_RESULT and ALERT_CHAT_ID:
        try:
            txt = "📢 <b>Novo resultado detectado no histórico!</b>\n" + " ".join(f"{d:02d}" for d in dezenas)
            await context.bot.send_message(chat_id=int(ALERT_CHAT_ID), text=txt, parse_mode="HTML")
        except Exception:
            pass

    # treino automático (se habilitado)
    learn = LearningCore()
    batch = _load_last_batch()
    apostas = batch.get("apostas") or []
    if AUTO_TRAIN_ON_NEW_RESULT and apostas:
        try:
            rel = learn.aprender_com_lote(oficial=dezenas, apostas=apostas, tag="auto_train")
            _append_audit({"event": "auto_train", "key": key, "resultado": dezenas, "rel": rel})
        except Exception as e:
            _append_audit({"event": "auto_train_error", "key": key, "err": str(e)})
    else:
        # mesmo sem treino, registramos o oficial para alimentar stats/histórico
        learn.registrar_resultado_oficial(dezenas)
        _append_audit({"event": "new_result_no_train", "key": key, "resultado": dezenas})


# ------------------------ Segurança / Avisos / Bloqueios ------------------------

# Avisos por usuário (3 avisos → bloqueio)
WARNINGS: Dict[int, int] = {}
# Usuários bloqueados (após 3 avisos)
BLOCKED_USERS: Set[int] = set()

# Anti-flood simples (por user + comando)
_last_call_per_user: Dict[tuple[int, str], float] = {}
COOLDOWN_SECONDS = 8.0


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
    """
    Retorna True se o usuário é administrador.
    Admin é definido fixo em ADMIN_IDS.
    """
    return user_id in ADMIN_IDS


def _is_blocked(user_id: int) -> bool:
    return user_id in BLOCKED_USERS


def _usuario_autorizado(user_id: int) -> bool:
    """
    Autorização geral para uso do bot:
    - Usuário NÃO pode estar bloqueado.
    - Admin sempre é autorizado.
    - Se WHITELIST_IDS estiver vazia → modo aberto (teste).
    - Se WHITELIST_IDS tiver IDs → só quem estiver nela pode usar.
    """
    # Bloqueado nunca entra
    if _is_blocked(user_id):
        return False

    # Admin sempre pode
    if _is_admin(user_id):
        return True

    # Se whitelist estiver vazia, consideramos modo aberto (desenvolvimento/teste)
    if not WHITELIST_IDS:
        return True

    # Em produção: somente IDs presentes na whitelist.txt
    return user_id in WHITELIST_IDS


async def _registrar_infracao(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Registra 1 infração quando o usuário envia algo que não deve:
    - Texto aleatório (sem comando)
    - Foto, documento, áudio, sticker, etc.
    Regras:
    - Admin NUNCA recebe infração.
    - Usuário comum recebe até 3 avisos; no 3º é bloqueado.
    """
    user = update.effective_user
    msg = update.message

    if user is None or msg is None:
        return

    user_id = user.id

    # Admin nunca leva strike
    if _is_admin(user_id):
        return

    # Se já estiver bloqueado, só avisa
    if _is_blocked(user_id):
        await msg.reply_text("🚫 Você está bloqueado. Entre em contato com o administrador.")
        return

    # Incrementa aviso
    WARNINGS[user_id] = WARNINGS.get(user_id, 0) + 1
    avisos = WARNINGS[user_id]

    if avisos < 3:
        await msg.reply_text(
            f"⚠️ Aviso {avisos}/3:\n"
            "Este bot aceita apenas comandos válidos (ex.: /gerar).\n"
            "Mensagens de texto, fotos, áudios ou outros envios fora do padrão não são permitidos.\n"
            "Após 3 avisos, seu acesso será bloqueado."
        )
    else:
        # Bloqueia usuário
        BLOCKED_USERS.add(user_id)
        await msg.reply_text(
            "🚫 Seu acesso ao bot foi BLOQUEADO por uso indevido (3 avisos).\n"
            "Apenas o administrador pode reverter esse bloqueio."
        )
        logger.warning(f"Usuário {user_id} bloqueado por uso indevido.")


# ------------------------ Helpers de histórico ------------------------

def carregar_historico(path: str) -> List[List[int]]:
    """
    Carrega um history.csv no formato *simples* (como o seu):

    - Cada linha: 15 dezenas separadas por vírgula (pode ter espaços).
      Ex.: 2,3,4,6,7,8,9,11,16,17,20,21,23,24,25
    - Ignora tudo depois de ';' (proteção contra anotações acidentais).
    - Se a primeira linha não tiver 15 dezenas válidas (1..25), trata como cabeçalho.
    - IMPORTANTE: o seu arquivo está em ordem "mais recente primeiro".
    """
    import os

    if not os.path.exists(path):
        return []

    def parse_line(line: str) -> List[int]:
        line = (line or "").split(";", 1)[0].strip()
        if not line:
            return []
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 15:
            return []
        try:
            nums = [int(x) for x in parts[-15:]]
        except Exception:
            return []
        nums = [n for n in nums if 1 <= n <= 25]
        if len(nums) != 15:
            return []
        # garante unicidade
        nums = sorted(set(nums))
        return nums if len(nums) == 15 else []

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    if not raw_lines:
        return []

    # cabeçalho?
    start_idx = 0
    if len(parse_line(raw_lines[0])) != 15:
        start_idx = 1

    hist: List[List[int]] = []
    for ln in raw_lines[start_idx:]:
        dezenas = parse_line(ln)
        if len(dezenas) == 15:
            hist.append(dezenas)

    return hist


def ultimo_resultado(historico: List[List[int]]) -> List[int]:
    """
    Considera que o arquivo está com o ÚLTIMO concurso na PRIMEIRA linha de dados.
    Ou seja: historico[0] = último resultado real.
    """
    if not historico:
        return []
    # já vem ordenado em carregar_historico, mas mantemos para garantir
    return list(sorted(historico[0]))


# ---------------------------- Comandos ----------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0

    if _is_admin(uid):
        msg = (
            "👋 Bem-vindo ao <b>LotoFácil Oráculo Supremo</b>.\n\n"
            "Comandos principais:\n"
            "/gerar – gera suas apostas Mestre com base no último resultado do histórico.\n"
            "/confirmar &lt;15 dezenas&gt; – aplica aprendizado sobre o último lote gerado (ADMIN).\n"
            "/desbloquear &lt;id&gt; – remove bloqueio de um usuário (ADMIN).\n"
            "/lista_bloqueados – lista todos os usuários bloqueados (ADMIN).\n"
            "/debug_state – mostra alpha, viés numérico e tamanho da memória (ADMIN).\n"
            "/meuid – mostra seu ID.\n\n"
            "Use com responsabilidade."
        )
    else:
        msg = (
            "👋 Bem-vindo ao <b>LotoFácil Oráculo Supremo</b>.\n\n"
            "Comandos disponíveis para você:\n"
            "/gerar – gera suas apostas Mestre com base no último resultado do histórico.\n"
            "/meuid – mostra seu ID.\n\n"
            "Após a compra, envie o ID exibido em /meuid para o administrador liberar seu acesso.\n\n"
            "⚠️ Não envie mensagens de texto aleatórias, fotos, áudios ou outros tipos de mídia.\n"
            "O bot é focado apenas em comandos. Após 3 avisos, seu acesso será bloqueado."
        )

    await update.message.reply_text(msg, parse_mode="HTML")


async def meuid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id if user else 0
    await update.message.reply_text(f"Seu ID é: `{uid}`", parse_mode="Markdown")


async def gerar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Comando SUPREMO:
    - Carrega último resultado do histórico
    - Carrega estado de aprendizado (bias_num + alpha)
    - Gera EXACT 10 apostas conforme OraculoEngine (Preset Mestre)
    - Exibe shape + estatísticas
    """
    user = update.effective_user
    user_id = user.id if user else 0
    chat_id = update.effective_chat.id

    # Bloqueio global
    if _is_blocked(user_id):
        return await update.message.reply_text(
            "🚫 Seu acesso ao bot está bloqueado.\n"
            "Apenas o administrador pode reverter esse bloqueio."
        )

    # Autorização (whitelist + admin + bloqueio)
    if not _usuario_autorizado(user_id):
        return await update.message.reply_text(
            "⛔ Seu acesso ainda não está liberado.\n\n"
            "Use /meuid e envie o seu ID para o administrador após a confirmação de pagamento.\n"
            "Assim que seu ID for incluído na whitelist, o comando /gerar ficará disponível."
        )

    if _hit_cooldown(user_id, "gerar"):
        return await update.message.reply_text(
            "⏳ Aguarde alguns segundos antes de usar /gerar novamente."
        )

    # Mensagem de carregamento
    try:
        loading = await update.message.reply_text(
            "⏳ Gerando suas apostas Mestre...\n[░░░░░░░░░░] 0%"
        )
    except Exception:
        loading = None

    async def _set_progress(pct: float, etapa: str) -> None:
        if loading is None:
            return
        pct = max(0.0, min(1.0, float(pct)))
        total = 10
        filled = int(round(total * pct))
        bar = "▰" * filled + "▱" * (total - filled)
        txt = (
            "⏳ Gerando suas apostas Mestre…\n"
            f"[{bar}] {int(pct * 100)}%\n\n"
            f"Etapa: {etapa}"
        )
        try:
            await loading.edit_text(txt)
        except Exception:
            pass

    await _set_progress(0.15, "Carregando histórico...")

    historico = carregar_historico(HISTORY_PATH)
    if not historico:
        if loading is not None:
            try:
                await loading.edit_text("Erro: histórico vazio ou inválido.")
            except Exception:
                pass
        return await update.message.reply_text("Erro: histórico vazio ou inválido.")

    ultimo = ultimo_resultado(historico)
    if len(ultimo) != 15:
        if loading is not None:
            try:
                await loading.edit_text("Erro: último resultado inválido (precisa de 15 dezenas).")
            except Exception:
                pass
        return await update.message.reply_text("Erro: último resultado inválido (precisa de 15 dezenas).")

    await _set_progress(0.35, "Carregando núcleo de aprendizado...")

    # Carrega estado de aprendizado robusto
    learn_core = LearningCore()
    alpha = learn_core.get_alpha()
    bias_num = learn_core.get_bias_num()

    # Oráculo configurado: overlap=11, target_qtd=10 (Preset Mestre)
    cfg = EngineConfig(overlap_max=11, target_qtd=10)
    engine = OraculoEngine(config=cfg, bias_num=bias_num, alpha=alpha)

    await _set_progress(0.55, "Gerando lote de apostas...")

    try:
        # Sempre pedimos EXACT 10 apostas
        seed = deterministic_seed(ultimo, version="mestre_v2", salt="lotofacil")
        apostas = engine.gerar_lote(ultimo_resultado=ultimo, qtd=10, seed=seed)
    except Exception as e:
        logger.error("Erro no OraculoEngine.gerar_lote: %s", e, exc_info=True)
        if loading is not None:
            try:
                await loading.edit_text(f"Erro interno ao gerar apostas: {e}")
            except Exception:
                pass
        return await update.message.reply_text(f"Erro interno ao gerar apostas: {e}")

    if not apostas or len(apostas) != 10:
        if loading is not None:
            try:
                await loading.edit_text("Não foi possível gerar as 10 apostas Mestre dentro das regras.")
            except Exception:
                pass
        return await update.message.reply_text(
            "Não foi possível gerar as 10 apostas Mestre dentro das regras (shape + anti-overlap)."
        )

    await _set_progress(0.75, "Calculando telemetria...")

    # Telemetria local com base no último resultado (apenas informativa)
    telems = []
    placares = []
    for a in apostas:
        hit = _hits(a, ultimo)
        placares.append(hit)
        pares, imp = paridade(a)
        seq = max_seq(a)
        R = len(set(a) & set(ultimo))
        telems.append((pares, imp, seq, R, hit))

    melhor = max(placares)
    media = sum(placares) / len(placares)

    # Registra lote como "gerado" (para histórico leve de aprendizado)
    learn_core.registrar_lote_gerado(oficial_base=ultimo, apostas=apostas, tag="gerar")

    # Salva lote e base em memória para o /confirmar
    global LAST_APOSTAS, LAST_BASE
    LAST_APOSTAS = [list(a) for a in apostas]
    LAST_BASE = list(ultimo)
    _save_last_batch(base=LAST_BASE, apostas=LAST_APOSTAS)
    _append_audit({"event": "gerar", "base": LAST_BASE, "seed": seed, "qtd": len(LAST_APOSTAS)})

    await _set_progress(0.95, "Formatando resposta...")

    # Monta resposta
    linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES — Preset Mestre</b> 🎰\n"]
    ok_count = 0
    for i, (a, (pares, imp, seq, R, hit)) in enumerate(zip(apostas, telems), start=1):
        status = "✅ OK" if (7 <= pares <= 8 and seq <= 3) else "🛠️ REVER"
        if status.startswith("✅"):
            ok_count += 1
        linhas.append(
            f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in a)}\n"
            f"🔢 Pares: {pares} | Ímpares: {imp} | SeqMax: {seq} | {R}R | "
            f"<i>{hit} acertos (vs. último)</i> | {status}\n"
        )

    linhas.append(
        f"\n📊 <b>Resumo do Lote</b>\n"
        f"• Melhor aposta (vs. último): <b>{melhor}</b> acertos\n"
        f"• Média do lote (vs. último): <b>{media:.2f}</b> acertos\n"
        f"• Conformidade shape Mestre: <b>{ok_count}/{len(apostas)}</b> dentro de (paridade 7–8, seq≤3)"
    )
    linhas.append(
        f"• Alpha atual do núcleo: <b>{alpha:.3f}</b> (usado apenas como viés de repetição)\n"
        f"• Lote ainda NÃO ajustou bias (aprendizado é feito depois, com o resultado oficial)."
    )

    now_sp = datetime.now(ZoneInfo(TIMEZONE))
    carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
    linhas.append(
        f"\n<i>base=último resultado | shape Mestre (7–8 pares, seq≤3) | "
        f"anti-overlap≤11 | tz={TIMEZONE} | {carimbo}</i>"
    )

    texto = "\n".join(linhas)

    if loading is not None:
        try:
            await loading.edit_text(texto, parse_mode="HTML")
        except Exception:
            await update.message.reply_text(texto, parse_mode="HTML")
    else:
        await update.message.reply_text(texto, parse_mode="HTML")


# --------------------------------------------------------
# /confirmar — apenas ADMIN, aplica aprendizado sobre o ÚLTIMO lote
# --------------------------------------------------------

async def confirmar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id if user else 0

    # Somente ADMIN
    if not _is_admin(user_id):
        return await update.message.reply_text("⛔ Este comando é restrito ao administrador.")

    # Anti flood
    if _hit_cooldown(user_id, "confirmar", cooldown=4.0):
        return await update.message.reply_text("⏳ Aguarde alguns segundos antes de usar /confirmar novamente.")

    texto = (update.message.text or "").strip().split()
    dezenas_raw = texto[1:]  # tudo após /confirmar

    # Validação das dezenas
    try:
        dezenas = [int(x) for x in dezenas_raw]
    except Exception:
        return await update.message.reply_text("Use: /confirmar <15 dezenas entre 1..25>")

    if len(dezenas) != 15 or any(d < 1 or d > 25 for d in dezenas):
        return await update.message.reply_text("❗ Envie exatamente 15 dezenas entre 1–25.")

    dezenas = sorted(dezenas)

    # Garante que existe um lote anterior (memória ou disco)
    global LAST_APOSTAS, LAST_BASE
    if not LAST_APOSTAS:
        batch = _load_last_batch()
        if batch.get("apostas"):
            LAST_APOSTAS = [list(a) for a in batch.get("apostas")]
            LAST_BASE = list(batch.get("base") or [])
    if not LAST_APOSTAS:
        return await update.message.reply_text(
            "⚠️ Ainda não há lote disponível (memória/disco).\n"
            "Use primeiro o comando /gerar para o bot ter apostas para analisar."
        )

    # Núcleo de aprendizado
    learn = LearningCore()
    alpha_before = learn.get_alpha()

    try:
        relatorio = learn.aprender_com_lote(
            oficial=dezenas,
            apostas=LAST_APOSTAS,
            tag="confirmar",
        )
    except Exception as e:
        logger.error(f"Erro no aprendizado em /confirmar: {e}", exc_info=True)
        return await update.message.reply_text(f"Erro interno no aprendizado: {e}")

    media = relatorio.get("media", 0.0)
    topk = relatorio.get("topk", 0.0)
    melhor = relatorio.get("melhor", 0)
    alpha_after = relatorio.get("alpha", alpha_before)
    placares = relatorio.get("placares", [])
    lote_bom = relatorio.get("lote_bom", False)

    # Monta relatório aposta a aposta
    linhas: List[str] = []

    linhas.append("✅ <b>Resultado analisado com sucesso!</b>\n")
    linhas.append(
        "• Resultado informado: <b>"
        + " ".join(f"{d:02d}" for d in dezenas)
        + "</b>\n"
    )

    linhas.append("<b>📊 Aprendizado aplicado sobre o ÚLTIMO lote gerado:</b>")
    linhas.append(f"• Melhor aposta: <b>{melhor}</b> acertos")
    linhas.append(f"• Média do lote: <b>{media:.2f}</b> acertos")
    linhas.append(f"• Top-K médio: <b>{topk:.2f}</b> acertos")
    linhas.append("")
    linhas.append(f"• Alpha antes: <b>{alpha_before:.3f}</b>")
    linhas.append(f"• Alpha depois: <b>{alpha_after:.3f}</b>")

    if lote_bom:
        linhas.append("• Qualificação do lote: <b>Lote forte</b> — reforço mais intenso aplicado nas dezenas-chave.")
    else:
        linhas.append("• Qualificação do lote: <b>Lote fraco</b> — ajuste suave, puxando bias em direção ao neutro.")

    linhas.append("\n🔍 <b>Desempenho aposta a aposta (vs. resultado informado):</b>")

    for i, aposta in enumerate(LAST_APOSTAS, start=1):
        hit = placares[i - 1] if i - 1 < len(placares) else _hits(aposta, dezenas)
        pares, imp = paridade(aposta)
        seq = max_seq(aposta)
        R = len(set(aposta) & set(dezenas))
        status = "✅ OK" if (7 <= pares <= 8 and seq <= 3) else "🛠️ REVER"

        linhas.append(
            f"<b>Aposta {i}:</b> "
            + " ".join(f"{n:02d}" for n in sorted(aposta))
        )
        linhas.append(
            f"   🔢 Pares: {pares} | Ímpares: {imp} | SeqMax: {seq} | {R}R | "
            f"<i>{hit} acertos</i> | {status}"
        )

    # Base usada no último /gerar
    if LAST_BASE:
        base_txt = " ".join(f"{d:02d}" for d in LAST_BASE)
        linhas.append(
            f"\n<i>Base usada no último /gerar (resultado de referência do lote): {base_txt}</i>"
        )
    else:
        linhas.append(
            "\n<i>Base usada no último /gerar: não disponível (LAST_BASE vazio).</i>"
        )

    msg = "\n".join(linhas)

    await update.message.reply_text(msg, parse_mode="HTML")


# --------------------------------------------------------
# /desbloquear — ADMIN remove bloqueio de um usuário
# --------------------------------------------------------

async def desbloquear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id if user else 0

    if not _is_admin(user_id):
        return await update.message.reply_text("⛔ Este comando é restrito ao administrador.")

    texto = (update.message.text or "").strip().split()
    if len(texto) < 2 or not texto[1].isdigit():
        return await update.message.reply_text(
            "Use: /desbloquear <ID_DO_USUARIO>\n"
            "Exemplo: /desbloquear 123456789"
        )

    alvo_id = int(texto[1])

    # Remove bloqueio e avisos
    BLOCKED_USERS.discard(alvo_id)
    WARNINGS.pop(alvo_id, None)

    await update.message.reply_text(
        f"✅ Usuário {alvo_id} foi DESBLOQUEADO e contadores de aviso foram zerados."
    )


# --------------------------------------------------------
# /lista_bloqueados — ADMIN lista todos os usuários bloqueados
# --------------------------------------------------------

async def lista_bloqueados(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id if user else 0

    if not _is_admin(user_id):
        return await update.message.reply_text("⛔ Este comando é restrito ao administrador.")

    if not BLOCKED_USERS:
        return await update.message.reply_text("✅ Não há usuários bloqueados no momento.")

    linhas: List[str] = ["🚫 <b>Usuários bloqueados atualmente:</b>"]
    for uid in sorted(BLOCKED_USERS):
        avisos = WARNINGS.get(uid, 0)
        linhas.append(f"• ID: <code>{uid}</code> | Avisos: {avisos}")

    msg = "\n".join(linhas)
    await update.message.reply_text(msg, parse_mode="HTML")



# --------------------------------------------------------
# /status — mostra estado do bot (admin: completo, user: básico)
# --------------------------------------------------------

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id if user else 0

    learn = LearningCore()
    alpha = learn.get_alpha()
    bias_num = learn.get_bias_num() or {}
    last_seen = _read_last_seen_key()
    batch = _load_last_batch()

    linhas: List[str] = []
    linhas.append("📡 <b>STATUS — Lotofácil Oráculo (Mestre v2)</b>\n")
    linhas.append(f"• Auto-alert: <b>{'ON' if AUTO_ALERT_ON_NEW_RESULT else 'OFF'}</b>")
    linhas.append(f"• Auto-train: <b>{'ON' if AUTO_TRAIN_ON_NEW_RESULT else 'OFF'}</b>")
    linhas.append(f"• Alpha: <b>{alpha:.3f}</b>")
    linhas.append(f"• Último visto (key): <code>{last_seen or '—'}</code>")
    linhas.append(f"• Último lote salvo: <b>{'SIM' if batch.get('apostas') else 'NÃO'}</b>")

    if _is_admin(user_id) and bias_num:
        itens = sorted(bias_num.items(), key=lambda kv: kv[1], reverse=True)
        top_pos = itens[:5]
        top_neg = sorted(itens, key=lambda kv: kv[1])[:5]
        fmt = lambda lst: ", ".join(f"{int(d):02d}({v:+.3f})" for d, v in lst)
        linhas.append("\n<b>Top vieses +</b>: " + fmt(top_pos))
        linhas.append("<b>Top vieses -</b>: " + fmt(top_neg))

    await update.message.reply_text("\n".join(linhas), parse_mode="HTML")


# --------------------------------------------------------
# /debug_state — ADMIN: inspeciona estado de aprendizado
# --------------------------------------------------------

async def debug_state(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id if user else 0

    if not _is_admin(user_id):
        return await update.message.reply_text("⛔ Este comando é restrito ao administrador.")

    learn = LearningCore()
    alpha = learn.get_alpha()
    bias_num = learn.get_bias_num() or {}

    # Tamanho da memória (quantos lotes armazenados)
    try:
        memoria = getattr(learn, "state", None)
        history = getattr(memoria, "history", []) if memoria is not None else []
        mem_len = len(history)
    except Exception:
        mem_len = 0

    janela = getattr(getattr(learn, "cfg", None), "janela", None)
    state_path = getattr(learn, "path", LEARN_STATE_PATH)

    linhas: List[str] = []

    linhas.append("🧠 <b>DEBUG DO NÚCLEO DE APRENDIZADO</b>\n")
    linhas.append(f"• Arquivo de estado: <code>{state_path}</code>")
    linhas.append(f"• Alpha atual: <b>{alpha:.3f}</b>")

    if janela is not None:
        linhas.append(f"• Janela configurada: <b>{janela}</b> concursos")

    linhas.append(f"• Lotes armazenados em memória: <b>{mem_len}</b>")

    if bias_num:
        itens = sorted(bias_num.items(), key=lambda kv: kv[1], reverse=True)
        top_pos = itens[:5]
        top_neg = sorted(itens, key=lambda kv: kv[1])[:5]

        def fmt(lst: List[tuple[int, float]]) -> str:
            return ", ".join(f"{dez:02d}({valor:+.3f})" for dez, valor in lst)

        linhas.append("\n<b>Top vieses POSITIVOS (dezenas mais reforçadas):</b>")
        linhas.append(fmt(top_pos))

        linhas.append("\n<b>Top vieses NEGATIVOS (dezenas mais enfraquecidas):</b>")
        linhas.append(fmt(top_neg))
    else:
        linhas.append("\nNenhum viés numérico armazenado ainda (bias_num vazio).")

    msg = "\n".join(linhas)
    await update.message.reply_text(msg, parse_mode="HTML")


# --------------------------------------------------------
# Handler genérico para qualquer conteúdo não-comando
# (texto solto, foto, vídeo, documento, áudio, sticker, etc.)
# --------------------------------------------------------

async def anti_abuso_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _registrar_infracao(update, context)


# ---------------------------- bootstrap ----------------------------

def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "COLOQUE_SEU_TOKEN_AQUI":
        raise RuntimeError("Defina BOT_TOKEN no ambiente ou dentro do bot.py antes de rodar.")

    # Log de caminho de estado de aprendizado (para conferência)
    logger.info(f"LEARN_STATE_PATH em uso: {LEARN_STATE_PATH}")

    # Recarrega whitelist no início (caso o arquivo tenha mudado entre deploys)
    global WHITELIST_IDS
    WHITELIST_IDS = carregar_whitelist(WHITELIST_PATH)
    logger.info(f"Whitelist carregada: {sorted(WHITELIST_IDS)}")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("meuid", meuid))
    app.add_handler(CommandHandler("gerar", gerar))
    app.add_handler(CommandHandler("confirmar", confirmar))
    app.add_handler(CommandHandler("desbloquear", desbloquear))
    app.add_handler(CommandHandler("lista_bloqueados", lista_bloqueados))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("debug_state", debug_state))

    # Qualquer mensagem que NÃO seja comando cai aqui (segurança máxima)
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, anti_abuso_handler))

    # Job periódico: detecta resultado novo no history.csv e aplica ciclo auto-alert/auto-train
    try:
        app.job_queue.run_repeating(_alert_and_train_if_new_result, interval=RESULT_CHECK_INTERVAL_SEC, first=10)
    except Exception as e:
        logger.error("Falha ao agendar watcher de resultado: %s", e)

    logger.info("Bot iniciado. Aguardando comandos...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
