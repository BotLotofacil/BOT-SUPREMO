# bot.py
from __future__ import annotations
import logging
import os
from typing import List
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
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
HISTORY_PATH = os.environ.get("HISTORY_PATH", "history.csv")

# Lista de usuários autorizados (string separada por vírgula)
# Exemplo: ADMIN_IDS="123456789,987654321"
ADMIN_IDS_ENV = os.environ.get("ADMIN_IDS", "")
ADMIN_IDS = {int(x) for x in ADMIN_IDS_ENV.replace(" ", "").split(",") if x.isdigit()}


# ------------------------ Helpers de histórico ------------------------


def carregar_historico(path: str) -> List[List[int]]:
    """
    Carrega um history.csv simples:
    - Supõe cabeçalho na primeira linha
    - Cada linha tem pelo menos 15 dezenas nas 15 últimas colunas
    """
    import csv

    if not os.path.exists(path):
        return []
    hist = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return []
    # ignora cabeçalho
    for row in rows[1:]:
        if len(row) < 15:
            continue
        # pega as 15 últimas colunas como dezenas
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
    if not historico:
        return []
    return list(sorted(historico[-1]))


# ------------------------ Helpers de autorização ------------------------


def _usuario_autorizado(user_id: int) -> bool:
    """Somente IDs na lista de admin são autorizados. Ajuste se quiser outra regra."""
    if not ADMIN_IDS:
        # se não tiver admin configurado, libera para todos (cuidado!)
        return True
    return user_id in ADMIN_IDS


# Anti-flood simples (por user + comando)
_last_call_per_user: dict[tuple[int, str], float] = {}
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


# ---------------------------- Comandos ----------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id if update.effective_user else 0
    msg = (
        "👋 Bem-vindo ao *LotoFácil Oráculo Supremo*.\n\n"
        "Comando principal:\n"
        "`/gerar` – gera suas apostas Mestre com base no último resultado.\n\n"
        "Use `/meuid` para ver seu ID e configurar autorização."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def meuid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id if update.effective_user else 0
    await update.message.reply_text(f"Seu ID é: `{uid}`", parse_mode="Markdown")


async def gerar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Comando SUPREMO:
    - Carrega último resultado do histórico
    - Carrega estado de aprendizado (bias_num + alpha)
    - Gera 15 apostas conforme OraculoEngine
    - Exibe shape + estatísticas
    """
    user = update.effective_user
    user_id = user.id if user else 0
    chat_id = update.effective_chat.id

    if not _usuario_autorizado(user_id):
        return await update.message.reply_text("⛔ Você não está autorizado a usar este bot.")

    if _hit_cooldown(user_id, "gerar"):
        return await update.message.reply_text(
            f"⏳ Aguarde alguns segundos antes de usar /gerar novamente."
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

    # Oráculo configurado
    cfg = EngineConfig(overlap_max=11, target_qtd=15)
    engine = OraculoEngine(config=cfg, bias_num=bias_num, alpha=alpha)

    await _set_progress(0.55, "Gerando lote de apostas...")

    try:
        apostas = engine.gerar_lote(ultimo_resultado=ultimo, qtd=15)
    except Exception as e:
        logger.error("Erro no OraculoEngine.gerar_lote: %s", e, exc_info=True)
        if loading is not None:
            try:
                await loading.edit_text(f"Erro interno ao gerar apostas: {e}")
            except Exception:
                pass
        return await update.message.reply_text(f"Erro interno ao gerar apostas: {e}")

    if not apostas:
        if loading is not None:
            try:
                await loading.edit_text("Não foi possível gerar apostas válidas.")
            except Exception:
                pass
        return await update.message.reply_text("Não foi possível gerar apostas válidas.")

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

    # Registra lote como "gerado" (para histórico)
    learn_core.registrar_lote_gerado(oficial_base=ultimo, apostas=apostas, tag="gerar")

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


# ---------------------------- bootstrap ----------------------------


def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "COLOQUE_SEU_TOKEN_AQUI":
        raise RuntimeError("Defina BOT_TOKEN no ambiente ou dentro do bot.py antes de rodar.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("meuid", meuid))
    app.add_handler(CommandHandler("gerar", gerar))

    logger.info("Bot iniciado. Aguardando comandos...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
