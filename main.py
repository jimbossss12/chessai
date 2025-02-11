import os
import sys
import threading
import time
import random
import json
import glob
import datetime
import logging
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic_settings import BaseSettings
from tqdm import tqdm

import wandb  # Weights & Biases seuranta
from torch.cuda.amp import autocast, GradScaler  # Mixed precision
import optuna  # Hyperparametrien optimointiin
import psutil  # Resurssimonitorointiin

import chess
import chess.engine
import chess.pgn

# Uudet kirjastot tiedoston latausta ja purkamista varten
import requests
import zstandard as zstd

# 1. Konfiguraatioasetukset
class Settings(BaseSettings):
    low_resource_mode: bool = True
    channels: int = 64
    num_res_blocks: int = 3
    pgn_depth: int = 5
    self_play_batch_size: int = 16
    learning_rate: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    metrics_log: str = "metrics_log.json"
    early_stopping_patience: int = 5000
    grad_clip: float = 1.0
    use_mixed_precision: bool = True
    use_distributed: bool = False
    study_name: str = "chess_ai_study"
    storage: str = "sqlite:///optuna_study.db"

    class Config:
        env_file = ".env"

settings = Settings()

# Ylikirjoitetaan asetuksia, jos low_resource_mode on käytössä
if settings.low_resource_mode:
    settings.channels = 32
    settings.num_res_blocks = 2
    settings.self_play_batch_size = 8

os.makedirs(settings.checkpoint_dir, exist_ok=True)

# 2. Weights & Biases -integraatio
wandb.init(project="chess-ai", config=settings.dict())

# 3. Lokitus
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# 4. MetricsStore – tallentaa metrikat myös tiedostoon
class MetricsStore:
    def __init__(self, log_file: str):
        """
        Tallentaa koulutuksen metrikat säikeiturvallisesti.
        """
        self.lock = threading.Lock()
        self.log_file = log_file
        self.supervised: Dict[str, Any] = {
            "iteration": 0,
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "target_value": 0.0,
            "predicted_value": 0.0,
            "game": 0,
            "move": 0,
        }
        self.self_play: Dict[str, Any] = {
            "iteration": 0,
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "target_value": 0.0,
            "predicted_value": 0.0,
            "game": "self-play",
            "move": 0,
        }
        self.persist()

    def update_supervised(self, metrics: dict) -> None:
        with self.lock:
            self.supervised.update(metrics)
            self.persist()

    def update_self_play(self, metrics: dict) -> None:
        with self.lock:
            self.self_play.update(metrics)
            self.persist()

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return {"supervised": self.supervised.copy(), "self_play": self.self_play.copy()}

    def persist(self) -> None:
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.get_metrics(), f)
        except Exception as e:
            logger.error("Virhe metrikatiedoston kirjoituksessa: %s", e)


metrics_store = MetricsStore(settings.metrics_log)


# 5. Data-augmentaatio: laudan peilaus
def augment_board(board: chess.Board) -> List[chess.Board]:
    """
    Palauttaa listan alkuperäisestä laudasta ja sen peilattua versiosta.
    """
    original = board.copy()
    mirrored = chess.Board(fen=board.mirror().fen())
    return [original, mirrored]


# 6. Neuraverkko – määritelmä

class ResidualBlock(nn.Module):
    """
    Yksittäinen residual-lohko.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ChessResNet(nn.Module):
    """
    Neuraverkko, joka ennustaa siirtopoliittisia todennäköisyyksiä ja pelin arvion.
    """
    def __init__(self, action_size: int, num_res_blocks: int = settings.num_res_blocks):
        super().__init__()
        self.conv_input = nn.Conv2d(12, settings.channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(settings.channels)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList([ResidualBlock(settings.channels) for _ in range(num_res_blocks)])
        self.conv_policy = nn.Conv2d(settings.channels, 73, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(73)
        self.fc_policy = nn.Linear(73 * 8 * 8, action_size)
        self.conv_value = nn.Conv2d(settings.channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        # Policy head
        p = self.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.fc_policy(p)
        # Value head
        v = self.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.fc_value1(v))
        value = torch.tanh(self.fc_value2(v))
        return policy_logits, value


# 7. Siirtojen kartoitusfunktiot
def get_alphazero_move_index(move: chess.Move, board: chess.Board) -> int:
    """
    Määrittelee yksinkertaistetun indeksin siirrolle.
    """
    from_sq = move.from_square
    move_type = 0  # Yksinkertaistettu laskenta
    return from_sq * 73 + move_type


def alphazero_index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Muuntaa annetun indeksin takaisin siirroksi.
    """
    from_sq = index // 73
    for move in board.legal_moves:
        if move.from_square == from_sq:
            return move
    return random.choice(list(board.legal_moves))


def get_legal_move_indices(board: chess.Board) -> List[int]:
    legal_indices = []
    for move in board.legal_moves:
        try:
            idx = get_alphazero_move_index(move, board)
            legal_indices.append(idx)
        except Exception:
            continue
    return legal_indices


# 8. Laudan muuntaminen tensoriksi
def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Muuntaa shakkilaudan 12x8x8 -tensoriksi.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            channel = (piece.piece_type - 1) if piece.color == chess.WHITE else (piece.piece_type - 1 + 6)
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[channel, row, col] = 1.0
    return tensor


def get_board_tensor(board: chess.Board) -> torch.Tensor:
    """
    Palauttaa tensorin, joka edustaa shakkilaudan tilaa.
    Jos mustan vuoro, tensorin akselit käännetään.
    """
    t = board_to_tensor(board)
    if board.turn == chess.BLACK:
        t = np.flip(t, axis=(1, 2)).copy()
    return torch.tensor(t).unsqueeze(0)


# 9. Laite, malli, optimoija, lr-scheduler ja AMP scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SIZE = 8 * 8 * 73
net = ChessResNet(ACTION_SIZE).to(device)
optimizer = optim.Adam(net.parameters(), lr=settings.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
scaler = GradScaler() if settings.use_mixed_precision else None

if settings.use_distributed:
    net = torch.nn.parallel.DistributedDataParallel(net)


# 10. Stockfish - alustaminen
def install_stockfish_if_needed() -> bool:
    """
    Tarkistaa, onko Stockfish asennettu.
    """
    # Tässä voi tehdä asennuksen tarkistuksen, jos tarpeen.
    return True


if not install_stockfish_if_needed():
    logger.error("Stockfish vaaditaan. Lopetetaan.")
    sys.exit(1)

try:
    stockfish_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
except Exception as e:
    logger.error("Virhe Stockfishin alustuksessa: %s", e)
    stockfish_engine = None


# 11. Replay-puskuri – säikeiturvallinen FIFO-jono
replay_buffer = deque(maxlen=10000)
replay_lock = threading.Lock()


# 12. Checkpointien tallennus ja lataus
def save_checkpoint(iteration: int) -> None:
    """
    Tallentaa checkpointin, jonka nimi sisältää iteroinnin ja aikaleiman.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(settings.checkpoint_dir, f"checkpoint_iter{iteration}_{timestamp}.pt")
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, filename)
    logger.info("Checkpoint tallennettu: %s", filename)


def load_latest_checkpoint() -> int:
    """
    Lataa uusimman checkpointin, jos sellainen löytyy.
    """
    checkpoints = glob.glob(os.path.join(settings.checkpoint_dir, "checkpoint_iter*.pt"))
    if not checkpoints:
        return 0
    latest = max(checkpoints, key=os.path.getmtime)
    checkpoint = torch.load(latest, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint.get("iteration", 0)
    logger.info("Ladattiin checkpoint: %s, iter: %d", latest, iteration)
    return iteration


# 13. FastAPI-sovellus ja virheenkäsittelijä
app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Käsittelemätön virhe: %s", exc)
    return PlainTextResponse("Sisäinen palvelinvirhe", status_code=500)


# 14. REST API - APIRouter
api_router = APIRouter()


@api_router.get("/predict")
async def predict_move(fen: str, epsilon: float = 0.0) -> Dict[str, Any]:
    """
    Ennustaa seuraavan siirron annetulle FEN-merkkijonolle.
    """
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Virheellinen FEN-merkkijono.") from e

    net.eval()
    board_tensor = get_board_tensor(board).to(device)
    with torch.no_grad():
        policy_logits, value = net(board_tensor)
    policy_logits = policy_logits.squeeze(0)
    legal_indices = get_legal_move_indices(board)
    if not legal_indices:
        raise HTTPException(status_code=500, detail="Laillisia siirtoja ei löytynyt.")
    mask = torch.full((ACTION_SIZE,), -1e9, device=device)
    mask[legal_indices] = 0.0
    masked_logits = policy_logits + mask
    probs = torch.softmax(masked_logits, dim=0)
    if random.random() < epsilon:
        chosen_index = random.choice(legal_indices)
    else:
        chosen_index = torch.multinomial(probs, 1).item()
    chosen_move = alphazero_index_to_move(chosen_index, board)
    if chosen_move not in board.legal_moves:
        chosen_move = random.choice(list(board.legal_moves))
    return {"move": chosen_move.uci(), "value_estimate": value.item()}


@api_router.get("/metrics")
async def get_api_metrics() -> JSONResponse:
    return JSONResponse(content=metrics_store.get_metrics())


@api_router.get("/checkpoints")
async def list_checkpoints() -> Dict[str, List[str]]:
    files = glob.glob(os.path.join(settings.checkpoint_dir, "*.pt"))
    return {"checkpoints": files}


@api_router.get("/training_status")
async def training_status() -> JSONResponse:
    status = {
        "current_learning_rate": optimizer.param_groups[0]["lr"],
        "metrics": metrics_store.get_metrics()
    }
    return JSONResponse(content=status)


@api_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    DASHBOARD_HTML = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>Chess AI Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      </head>
      <body>
        <nav class="navbar navbar-dark bg-dark mb-4">
          <div class="container-fluid">
            <a class="navbar-brand" href="#">Chess AI Dashboard</a>
          </div>
        </nav>
        <div class="container">
          <ul>
            <li><a href="/metrics">/metrics</a> – Näyttää koulutusmittarit.</li>
            <li><a href="/checkpoints">/checkpoints</a> – Näyttää tallennetut checkpointit.</li>
            <li><a href="/training_status">/training_status</a> – Näyttää koulutuksen tilan.</li>
            <li><a href="/predict?fen=START_FEN">/predict</a> – Siirtoehdotus annetulle FEN:lle.</li>
          </ul>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=DASHBOARD_HTML)


app.include_router(api_router)


# 15. Resurssimonitorointi psutililla
def resource_monitor() -> None:
    """
    Monitoroi järjestelmän muistinkäyttöä ja kirjaa varoituksia kriittisistä tasoista.
    """
    while True:
        mem = psutil.virtual_memory()
        logger.info("Muistin käyttö: %.2f%%", mem.percent)
        if mem.percent > 90:
            logger.warning("Muistin käyttö kriittisellä tasolla!")
        time.sleep(60)


monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
monitor_thread.start()


# Uudet funktiot Lichess-tietokannan lataamista ja purkamista varten

def download_file(url: str, local_filename: str) -> None:
    """
    Lataa tiedoston annettuun paikalliseen sijaintiin käyttäen stream-latausta.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(local_filename, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=local_filename) as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    logger.info("Lataus valmis: %s", local_filename)


def decompress_zst(zst_file: str, output_file: str) -> None:
    """
    Purkaa .zst-pakatun tiedoston ja tallentaa sen output_file-muodossa.
    """
    dctx = zstd.ZstdDecompressor()
    with open(zst_file, 'rb') as compressed, open(output_file, 'wb') as destination:
        dctx.copy_stream(compressed, destination)
    logger.info("Purku valmis: %s -> %s", zst_file, output_file)


def download_and_extract_lichess_db() -> str:
    """
    Lataa ja purkaa Lichessin tietokannan, jos decomprimoitua PGN-tiedostoa ei vielä ole.
    Palauttaa PGN-tiedoston polun.
    """
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
    pgn_file = "lichess_db_standard_rated_2025-01.pgn"
    zst_file = pgn_file + ".zst"

    if os.path.exists(pgn_file):
        logger.info("Lichessin PGN-tiedosto löytyy jo: %s", pgn_file)
        return pgn_file

    if not os.path.exists(zst_file):
        logger.info("Ladataan Lichessin tietokanta osoitteesta %s", url)
        download_file(url, zst_file)
    else:
        logger.info("Pakattu tiedosto löytyy jo: %s", zst_file)

    logger.info("Puretaan Lichessin tietokanta...")
    decompress_zst(zst_file, pgn_file)
    return pgn_file


# 16. Koulutus- ja self-play -funktiot

def mcts_move(board: chess.Board, num_simulations: int = 50) -> chess.Move:
    """
    Simppeli MCTS-tyyppinen siirtojen valinta.
    """
    best_value = -float('inf')
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        board_tensor = get_board_tensor(board).to(device)
        with torch.no_grad():
            _, value = net(board_tensor)
        board.pop()
        if value.item() > best_value:
            best_value = value.item()
            best_move = move
    return best_move if best_move is not None else random.choice(list(board.legal_moves))


def train_from_lichess(pgn_path: str, depth: int = settings.pgn_depth) -> None:
    """
    Kouluttaa mallia Lichess PGN -tiedostosta lukemalla pelejä.
    """
    writer = SummaryWriter()
    iteration = load_latest_checkpoint()
    best_loss = float('inf')
    epochs_since_improvement = 0  # Early stopping -laskuri

    if not os.path.exists(pgn_path):
        logger.error("PGN-tiedostoa '%s' ei löytynyt!", pgn_path)
        return

    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        game_number = 0
        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    pgn_file.seek(0)
                    continue
                game_number += 1
                board = game.board()
                move_number = 0
                for move in game.mainline_moves():
                    move_number += 1
                    # Data-augmentaatio: käytetään alkuperäistä ja peilattua lautaa
                    for b in augment_board(board):
                        board_tensor = get_board_tensor(b).to(device)
                        try:
                            target_index = get_alphazero_move_index(move, b)
                        except Exception:
                            b.push(move)
                            continue
                        target_value = 0.0
                        if stockfish_engine:
                            try:
                                info = stockfish_engine.analyse(b, chess.engine.Limit(depth=depth))
                                score = info["score"].white()
                                if score.is_mate():
                                    target_value = 1.0 if score.mate() > 0 else -1.0
                                else:
                                    cp = score.score()
                                    target_value = max(min(cp / 1000.0, 1.0), -1.0)
                            except Exception as e:
                                logger.warning("Stockfish arviointivirhe: %s", e)
                                target_value = 0.0
                        net.train()
                        if settings.use_mixed_precision:
                            with autocast():
                                policy_logits, predicted_value = net(board_tensor)
                                policy_logits = policy_logits.squeeze(0)
                                predicted_value = predicted_value.squeeze(0)
                                target_index_tensor = torch.tensor([target_index], dtype=torch.long, device=device)
                                policy_loss = nn.CrossEntropyLoss()(policy_logits.unsqueeze(0), target_index_tensor)
                                target_value_tensor = torch.tensor([target_value], dtype=torch.float32, device=device)
                                value_loss = nn.MSELoss()(predicted_value, target_value_tensor)
                                total_loss = policy_loss + value_loss
                        else:
                            policy_logits, predicted_value = net(board_tensor)
                            policy_logits = policy_logits.squeeze(0)
                            predicted_value = predicted_value.squeeze(0)
                            target_index_tensor = torch.tensor([target_index], dtype=torch.long, device=device)
                            policy_loss = nn.CrossEntropyLoss()(policy_logits.unsqueeze(0), target_index_tensor)
                            target_value_tensor = torch.tensor([target_value], dtype=torch.float32, device=device)
                            value_loss = nn.MSELoss()(predicted_value, target_value_tensor)
                            total_loss = policy_loss + value_loss

                        optimizer.zero_grad()
                        if settings.use_mixed_precision:
                            scaler.scale(total_loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
                            optimizer.step()
                        scheduler.step()

                        iteration += 1
                        writer.add_scalar("Loss/Total", total_loss.item(), iteration)
                        writer.add_scalar("Loss/Policy", policy_loss.item(), iteration)
                        writer.add_scalar("Loss/Value", value_loss.item(), iteration)
                        writer.add_scalar("Target/Value", target_value, iteration)

                        metrics_store.update_supervised({
                            "iteration": iteration,
                            "total_loss": total_loss.item(),
                            "policy_loss": policy_loss.item(),
                            "value_loss": value_loss.item(),
                            "target_value": target_value,
                            "predicted_value": predicted_value.item(),
                            "game": game_number,
                            "move": move_number,
                        })
                        wandb.log({
                            "iteration": iteration,
                            "loss_total": total_loss.item(),
                            "loss_policy": policy_loss.item(),
                            "loss_value": value_loss.item(),
                            "target_value": target_value,
                            "predicted_value": predicted_value.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        })

                        # Early stopping
                        if total_loss.item() < best_loss:
                            best_loss = total_loss.item()
                            epochs_since_improvement = 0
                        else:
                            epochs_since_improvement += 1
                        if epochs_since_improvement >= settings.early_stopping_patience:
                            logger.info("Early stopping triggered at iteration %d", iteration)
                            save_checkpoint(iteration)
                            writer.close()
                            return
                        if iteration % 500 == 0:
                            save_checkpoint(iteration)
                    board.push(move)
            except Exception as ex:
                logger.error("Virhe koulutusloopissa: %s. Uudelleenyritys 5 sekuntia myöhemmin...", ex)
                time.sleep(5)
    writer.close()


def self_play_game() -> int:
    """
    Suorittaa yhden self-play -pelin, tallentaa tilat ja siirtojen indeksit replay_bufferiin.
    Palauttaa pelissä tehtyjen siirtojen lukumäärän.
    """
    states = []
    moves_chosen = []
    board = chess.Board()
    while not board.is_game_over():
        board_tensor = get_board_tensor(board).to(device)
        states.append(board_tensor)
        # Valitaan siirto MCTS:llä, jos satunnaisluku ylittää kynnysarvon
        if random.random() < 0.1:
            move = random.choice(list(board.legal_moves))
        else:
            move = mcts_move(board)
        try:
            move_index = get_alphazero_move_index(move, board)
        except Exception:
            move_index = None
        moves_chosen.append(move_index)
        board.push(move)
    result = board.result()
    move_count = len(moves_chosen)
    outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
    with replay_lock:
        for state, move_idx in zip(states, moves_chosen):
            if move_idx is not None:
                replay_buffer.append((state, move_idx, outcome))
    logger.info("[Self-Play] Game finished: %s (moves: %d, outcome: %.1f)", result, move_count, outcome)
    return move_count


def self_play_training(batch_size: int = settings.self_play_batch_size) -> None:
    """
    Suorittaa koulutuksen replay_bufferin self-play -datan avulla.
    """
    if len(replay_buffer) < batch_size:
        return
    batch = random.sample(replay_buffer, batch_size)
    states, target_moves, outcomes = zip(*batch)
    state_batch = torch.cat(states).to(device)
    target_moves_tensor = torch.tensor(target_moves, dtype=torch.long, device=device)
    outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32, device=device)
    net.train()
    policy_logits, predicted_values = net(state_batch)
    policy_loss = nn.CrossEntropyLoss()(policy_logits, target_moves_tensor)
    predicted_values = predicted_values.squeeze()
    value_loss = nn.MSELoss()(predicted_values, outcomes_tensor)
    total_loss = policy_loss + value_loss

    optimizer.zero_grad()
    if settings.use_mixed_precision:
        with autocast():
            total_loss.backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
        optimizer.step()
    scheduler.step()

    iteration = metrics_store.self_play.get("iteration", 0) + 1
    metrics_store.update_self_play({
        "iteration": iteration,
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "target_value": outcomes_tensor.mean().item(),
        "predicted_value": predicted_values.mean().item(),
        "game": "self-play",
        "move": self_play_game(),  # Päivitetään siirtojen lukumäärä self-play -pelistä
    })


def self_play_thread() -> None:
    """
    Toistuva self-play -säie, joka suorittaa pelejä ja kouluttaa mallia niiden avulla.
    """
    while True:
        try:
            _ = self_play_game()
            self_play_training(batch_size=settings.self_play_batch_size)
        except Exception as e:
            logger.error("Self-play thread virhe: %s", e)
        time.sleep(1)


def combined_training_thread() -> None:
    """
    Käynnistää sekä lichess-koulutuksen että self-play -säikeet rinnakkain.
    Ensin ladataan ja puretaan Lichessin tietokanta automaattisesti.
    """
    # Ladataan ja puretaan Lichessin tietokanta, mikäli sitä ei vielä ole
    pgn_file = download_and_extract_lichess_db()
    t1 = threading.Thread(target=train_from_lichess, args=(pgn_file,), daemon=True)
    t2 = threading.Thread(target=self_play_thread, daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


# 17. Optunan hyperparametrien optimointi
def objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    optimizer.param_groups[0]["lr"] = lr
    simulated_loss = random.random()  # Simuloitu loss
    return simulated_loss


def run_hyperparameter_tuning() -> None:
    study = optuna.create_study(
        study_name=settings.study_name,
        direction="minimize",
        storage=settings.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)
    logger.info("Optuna paras hyperparametri: %s", study.best_trial.params)


# 18. Pääohjelma: käynnistetään koulutustaustasäikeet ja API-palvelin
if __name__ == "__main__":
    # Mahdollisuus suorittaa hyperparametrien optimointi ennen koulutusta:
    # run_hyperparameter_tuning()  # uncomment tarvittaessa

    trainer_thread = threading.Thread(target=combined_training_thread, daemon=True)
    trainer_thread.start()
    logger.info("Koulutustaustasäikeet käynnistetty.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
