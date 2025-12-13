import discord
from discord.ext import commands
import subprocess
import os
import random
import time
import matplotlib.pyplot as plt
from config import DISCORD_TOKEN, PROJECT_PATH, ALLOWED_SCRIPTS

LOG_FILE = "logs.txt"

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------- UTILIDADES ----------------------
def log_action(user, action):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.ctime()} | {user} | {action}\n")


def run_safe(cmd):
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_PATH
        )
        return result.stdout[:1800]
    except Exception as e:
        return f"Error: {e}"


# ---------------------- EVENTOS ----------------------
@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user}")


# ---------------------- COMANDO RUN ----------------------
@bot.command()
async def run(ctx, script: str):
    """
    Ejecutar RA1 o RA2:
    !run batch
    !run hidden
    !run epochs
    !run ra2
    """
    if script not in ALLOWED_SCRIPTS:
        await ctx.send("❌ Script no permitido.")
        return

    await ctx.send(f"⏳ Ejecutando **{script}**...")

    log_action(ctx.author, f"run {script}")

    output = run_safe(ALLOWED_SCRIPTS[script])
    await ctx.send(f"```\n{output}\n```")


GRAPH_MAP = {
    "batch": [
        "batch_size_accuracy.png",
        "batch_size_time.png"
    ],
    "hidden": [
        "hidden_size_accuracy.png",
        "hidden_size_time.png"
    ],
    "ra3": [
        "ra3_comparison.png"
    ],
    "topk": [
        "topk_comparison.png",
        "topk_methods_comparison.png"
    ]
}

@bot.command()
async def graph(ctx, tipo: str):
    """
    Ejemplos:
    !graph batch
    !graph hidden
    !graph ra3
    !graph topk
    """
    tipo = tipo.lower()

    if tipo not in GRAPH_MAP:
        await ctx.send("❌ Tipo de gráfico no válido.")
        return

    sent_any = False

    for filename in GRAPH_MAP[tipo]:
        path = os.path.join(PROJECT_PATH, "graphs", filename)
        if os.path.exists(path):
            await ctx.send(file=discord.File(path))
            sent_any = True

    if not sent_any:
        await ctx.send("❌ No se encontraron los gráficos esperados.")

@bot.command()
async def topk(ctx, n: int, k: int):
    """
    Ejecuta Sort, Heap, Quickselect usando random:
    !topk 20 5
    """
    arr = [random.randint(1, 999) for _ in range(n)]
    log_action(ctx.author, f"topk {n} {k}")

    # --- Sort ---
    start = time.time()
    sorted_res = sorted(arr)[-k:]
    sort_time = time.time() - start

    # --- Heap ---
    import heapq
    start = time.time()
    heap_res = heapq.nlargest(k, arr)
    heap_time = time.time() - start

    # --- Quickselect ---
    def quickselect(lst, k):
        if len(lst) <= k:
            return lst
        pivot = random.choice(lst)
        highs = [x for x in lst if x > pivot]
        lows = [x for x in lst if x <= pivot]
        if len(highs) == k:
            return highs
        if len(highs) > k:
            return quickselect(highs, k)
        return highs + quickselect(lows, k - len(highs))

    start = time.time()
    qselect_res = quickselect(arr, k)
    qselect_time = time.time() - start

    await ctx.send(
        f"🔢 **Top-K Resultados (n={n}, k={k})**\n"
        f"Sort: `{sorted_res}` ({sort_time:.6f}s)\n"
        f"Heap: `{heap_res}` ({heap_time:.6f}s)\n"
        f"Quickselect: `{qselect_res}` ({qselect_time:.6f}s)\n"
    )


# ---------------------- COMANDO STATUS ----------------------
@bot.command()
async def status(ctx):
    log_action(ctx.author, f"status request")

    await ctx.send(
        "📊 **Estado del Bot**\n"
        f"Proyecto: `{PROJECT_PATH}`\n"
        f"Scripts permitidos: {list(ALLOWED_SCRIPTS.keys())}\n"
        f"Logs: `{LOG_FILE}`\n"
        f"Dashboard local: http://localhost:5000 (si está activo)\n"
    )


# ---------------------- COMANDO LOGS ----------------------
@bot.command()
async def logs(ctx):
    if not os.path.exists(LOG_FILE):
        await ctx.send("No hay logs aún.")
        return

    with open(LOG_FILE, "r") as f:
        content = f.read()

    await ctx.send(f"```\n{content[:1900]}\n```")


#----------------------------------------------------------
@bot.command()
async def helpme(ctx):
    await ctx.send(
        "📘 **Comandos disponibles**\n\n"
        "**Ejecución de scripts:**\n"
        "`!run batch` → RA1 Batch Size\n"
        "`!run hidden` → RA1 Hidden Size\n"
        "`!run epochs` → RA1 Epochs\n"
        "`!run ra2` → RA2 Top-K Experimento\n"
        "`!run ra3` → RA3 Estructuras\n\n"
        "**Top-K interactivo:**\n"
        "`!topk n k` → ejemplo: `!topk 20 5`\n\n"
        "**Gráficos:**\n"
        "`!graph batch | hidden | ra3 | topk`\n"
    )

# ---------------------- INICIAR BOT ----------------------
bot.run(DISCORD_TOKEN)
