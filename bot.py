from __future__ import annotations
import logging
import os
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

from engine_oraculo import OraculoEngine, EngineConfig, shape_ok_mestre, paridade, max_seq
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

# Lista de usuários administradores (string separada por vírgula)
# Exemplo: ADMIN_IDS="123456789,987654321"
ADMIN_IDS_ENV = os.environ.get("ADMIN_IDS", "")
ADMIN_IDS: Set[int] = {
    int(x) for x in ADMIN_IDS_ENV.replace(" ", "").split(",") if x.isdigit()
}

# ------------------------ Estado em memória (/gerar -> /confirmar) ------------------------

# Lote mais recente gerado pelo /gerar
LAST_APOSTAS: List[List[int]] = []
# Base (último resultado do history) usada para gerar esse lote
LAST_BASE: List[int] = []

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
    - Se ADMIN_IDS estiver vazio, considera todos como admin (modo desenvolvimento).
    - Em produção, configure ADMIN_IDS com seu ID para restringir.
    """
    if not ADMIN_IDS:
        return True
    return user_id in ADMIN_IDS


def _is_blocked(user_id: int) -> bool:
    return user_id in BLOCKED_USERS


def _usuario_autorizado(user_id: int) -> bool:
    """
    Autorização geral para uso do bot:
    - Usuário NÃO pode estar bloqueado.
    - Não exige ser admin (para /gerar).
    """
    if _is_blocked(user_id):
        return False
    return True


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
    Carrega um history.csv simples, sem assumir forma fixa:
    - Se a primeira linha NÃO for dezenas válidas (1–25), trata como cabeçalho
    - Caso contrário, considera que já é um resultado
    - Cada linha deve ter pelo menos 15 colunas; usamos sempre as 15 últimas
    """
    import csv
    import os

    if not os.path.exists(path):
        return []

    hist: List[List[int]] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]  # ignora linhas totalmente vazias

    if not rows:
        return []

    def linha_eh_dezenas(row: List[str]) -> bool:
        """Retorna True se a linha parecer ser um resultado válido da Lotofácil."""
        if len(row) < 15:
            return False
        dezenas_raw = row[-15:]
        try:
            dezenas = [int(x) for x in dezenas_raw]
        except Exception:
            return False
        return all(1 <= d <= 25 for d in dezenas)

    # Detecta se a primeira linha é cabeçalho ou já é um resultado
    start_idx = 0
    if not linha_eh_dezenas(rows[0]):
        # primeira linha é cabeçalho → começamos da linha 2
        start_idx = 1

    for row in rows[start_idx:]:
        if len(row) < 15:
            continue
        dezenas_raw = row[-15:]
        try:
            dezenas = [int(x) for x in dezenas_raw]
        except Exception:
            continue
        dezenas = [d for d in dezenas if 1 <= d <= 25]
        if len(dezenas) == 15:
            dezenas = sorted(dezenas)
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

    # Mensagem diferente para admin x usuário comum
    if _is_admin(uid):
        msg = (
            "👋 Bem-vindo ao *LotoFácil Oráculo Supremo*.\n\n"
            "Comandos principais:\n"
            "/gerar – gera suas apostas Mestre com base no último resultado do histórico.\n"
            "/confirmar <15 dezenas> – aplica aprendizado sobre o último lote gerado (ADMIN).\n"
            "/desbloquear <id> – remove bloqueio de um usuário (ADMIN).\n"
            "/meuid – mostra seu ID.\n\n"
            "Use com responsabilidade."
        )
    else:
        msg = (
            "👋 Bem-vindo ao *LotoFácil Oráculo Supremo*.\n\n"
            "Comandos disponíveis para você:\n"
            "/gerar – gera suas apostas Mestre com base no último resultado do histórico.\n"
            "/meuid – mostra seu ID.\n\n"
            "⚠️ Não envie mensagens de texto aleatórias, fotos, áudios ou outros tipos de mídia.\n"
            "O bot é focado apenas em comandos. Após 3 avisos, seu acesso será bloqueado."
        )

    await update.message.reply_text(msg, parse_mode="Markdown")


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

    # Para /gerar, não exigimos admin — apenas não pode estar bloqueado
    if not _usuario_autorizado(user_id):
        return await update.message.reply_text("⛔ Você não está autorizado a usar este bot.")

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
        apostas = engine.gerar_lote(ultimo_resultado=ultimo, qtd=10)
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

    # Garante que existe um lote anterior
    if not LAST_APOSTAS:
        return await update.message.reply_text(
            "⚠️ Ainda não há lote em memória.\n"
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
# Handler genérico para qualquer conteúdo não-comando
# (texto solto, foto, vídeo, documento, áudio, sticker, etc.)
# --------------------------------------------------------

async def anti_abuso_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _registrar_infracao(update, context)


# ---------------------------- bootstrap ----------------------------

def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "COLOQUE_SEU_TOKEN_AQUI":
        raise RuntimeError("Defina BOT_TOKEN no ambiente ou dentro do bot.py antes de rodar.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("meuid", meuid))
    app.add_handler(CommandHandler("gerar", gerar))
    app.add_handler(CommandHandler("confirmar", confirmar))
    app.add_handler(CommandHandler("desbloquear", desbloquear))

    # Qualquer mensagem que NÃO seja comando cai aqui (segurança máxima)
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, anti_abuso_handler))

    logger.info("Bot iniciado. Aguardando comandos...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
