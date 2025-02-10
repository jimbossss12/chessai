#!/usr/bin/env python3
"""
Chess AI with Automated Installation of Stockfish, Lichess October 2020 Database,
Supervised Training from PGN, Self–Play Reinforcement Learning, and a Modern REST API Dashboard.

Features and improvements:
  - Automated installation of Stockfish and PGN database download/decompression.
  - Supervised training using Lichess PGN data with Stockfish evaluations.
  - Self–Play with a simple MCTS and replay buffer for reinforcement learning.
  - Option for an AlphaZero–style move mapping (8×8×73 output channels) for the policy head.
  - REST API endpoints for predicting moves, retrieving training metrics,
    simulating a game, and using a simplified MCTS-based move selection.
  - Modern dashboard (using Bootstrap 5) that shows separate training metrics for supervised and self–play modes.
  - Low resource mode: Smaller network and lower training parameters for systems with limited resources.

Dependencies:
    pip install python-chess torch tensorboard fastapi uvicorn requests zstandard tqdm

Run as:
    python main.py
"""

import os
import sys
import subprocess
import shutil
import requests
import zstandard as zstd  # For decompressing .zst files
import threading
import time
import random
from collections import deque
import logging

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from tqdm import tqdm  # For progress bars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

#############################################
# Global Configuration Flags
#############################################
LOW_RESOURCE_MODE = True  # Aseta True, jos käytössä on pienemmät järjestelmäresurssit
USE_ADVANCED_MOVE_MAPPING = True  # Käytetäänkö AlphaZero–tyylistä siirtojen esitystapaa
USE_SELF_PLAY = True  # Käytetäänkö itsensäpelausta ja vahvistusoppimista

if LOW_RESOURCE_MODE:
    CHANNELS = 64            # Pienempi määrä suodattimia
    NUM_RES_BLOCKS = 3       # Vähemmän residual–blokkeja
    PGN_DEPTH = 5            # Matalampi Stockfish–haun syvyys
    SELF_PLAY_BATCH_SIZE = 16  # Pienemmät eräkoot self–play koulutuksessa
    torch.set_num_threads(2)
else:
    CHANNELS = 256
    NUM_RES_BLOCKS = 5
    PGN_DEPTH = 10
    SELF_PLAY_BATCH_SIZE = 32

# Määritellään siirtojen esitystapa:
if not USE_ADVANCED_MOVE_MAPPING:
    def generate_move_mapping():
        moves_set = set()
        for source in range(64):
            for target in range(64):
                move = chess.Move(source, target)
                moves_set.add(move.uci())
                sr = chess.square_rank(source)
                tr = chess.square_rank(target)
                if sr == 6 and tr == 7:
                    for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        moves_set.add(chess.Move(source, target, promotion=prom).uci())
                if sr == 1 and tr == 0:
                    for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        moves_set.add(chess.Move(source, target, promotion=prom).uci())
        moves_list = sorted(list(moves_set))
        move_to_index = {m: i for i, m in enumerate(moves_list)}
        index_to_move = {i: m for i, m in enumerate(moves_list)}
        return moves_list, move_to_index, index_to_move

    moves_list, move_to_index, index_to_move = generate_move_mapping()
    ACTION_SIZE = len(moves_list)
    logging.info("Action space size (original mapping): %d", ACTION_SIZE)
else:
    # AlphaZero–tyylistä mappingia käytetään: 8×8×73 = 4672 ulostuloa
    ACTION_SIZE = 8 * 8 * 73
    logging.info("Action space size (AlphaZero mapping): %d", ACTION_SIZE)

#############################################
# 0. Helper Functions for Installation & Download
#############################################
def install_stockfish_if_needed():
    """Tarkistaa, onko Stockfish asennettu. Jos ei, asennetaan apt-get:lla."""
    stockfish_path = shutil.which("stockfish")
    if stockfish_path:
        logging.info("Stockfish found at: %s", stockfish_path)
        return True
    else:
        logging.info("Stockfish not found. Attempting to install via apt-get...")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "stockfish"], check=True)
            stockfish_path = shutil.which("stockfish")
            if stockfish_path:
                logging.info("Stockfish installed at: %s", stockfish_path)
                return True
            else:
                logging.error("Stockfish installation failed.")
                return False
        except Exception as e:
            logging.error("Error installing Stockfish: %s", e)
            return False

def download_lichess_database():
    """Lataa ja purkaa Lichessin PGN–tietokanta (October 2020) mikäli sitä ei löydy."""
    pgn_file = "lichess_db_standard_rated_2020-10.pgn"
    compressed_file = pgn_file + ".zst"
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2020-10.pgn.zst"
    
    if os.path.exists(pgn_file):
        logging.info("PGN file '%s' already exists.", pgn_file)
        return pgn_file

    if not os.path.exists(compressed_file):
        logging.info("Downloading Lichess database from %s ...", url)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            with open(compressed_file, "wb") as f_out, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc="Downloading PGN"
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f_out.write(chunk)
                        progress_bar.update(len(chunk))
            logging.info("Downloaded compressed PGN file to %s", compressed_file)
        except Exception as e:
            logging.error("Error downloading Lichess database: %s", e)
            sys.exit(1)

    logging.info("Decompressing the PGN file...")
    try:
        dctx = zstd.ZstdDecompressor()
        compressed_size = os.path.getsize(compressed_file)
        with open(compressed_file, 'rb') as compressed, open(pgn_file, 'wb') as out_file, tqdm(
            total=compressed_size, unit='B', unit_scale=True, desc="Decompressing PGN"
        ) as progress_bar:
            for chunk in dctx.read_to_iter(compressed, read_size=8192):
                out_file.write(chunk)
                progress_bar.update(len(chunk))
        logging.info("Decompressed PGN saved as '%s'", pgn_file)
    except Exception as e:
        logging.error("Error decompressing PGN file: %s", e)
        sys.exit(1)
    
    return pgn_file

#############################################
# 1. AlphaZero–tyylinen siirtojen esitystapa (jos käytössä)
#############################################
if USE_ADVANCED_MOVE_MAPPING:
    def get_alphazero_move_index(move, board):
        """
        Laskee annetulle siirrolle indeksin: from_square * 73 + move_type.
        Tämä funktio erottaa liukuvat siirrot, ratsun ja korotukset (capture/non-capture erotellaan).
        """
        from_sq = move.from_square
        to_sq = move.to_square
        dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
        dc = chess.square_file(to_sq) - chess.square_file(from_sq)
        knight_offsets = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
        if (dr, dc) in knight_offsets:
            knight_index = knight_offsets.index((dr, dc))
            move_type = 56 + knight_index  # indeksit 56-63
        elif move.promotion is not None:
            promotion_order = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
            if board.is_capture(move):
                move_type = 64 + promotion_order.get(move.promotion, 0)
            else:
                move_type = 68 + promotion_order.get(move.promotion, 0)
        else:
            directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
            found = False
            for d_idx, (dr_dir, dc_dir) in enumerate(directions):
                # Yksinkertaistettu: etsitään askelmäärä
                if dr_dir != 0 and dc_dir != 0:
                    if (dr % dr_dir == 0) and (dc % dc_dir == 0) and ((dr // dr_dir) == (dc // dc_dir)) and ((dr // dr_dir) > 0):
                        steps = dr // dr_dir
                        found = True
                        break
                elif dr_dir == 0 and dc_dir != 0:
                    if dr == 0 and (dc % dc_dir == 0) and (dc // dc_dir > 0):
                        steps = dc // dc_dir
                        found = True
                        break
                elif dc_dir == 0 and dr_dir != 0:
                    if dc == 0 and (dr % dr_dir == 0) and (dr // dr_dir > 0):
                        steps = dr // dr_dir
                        found = True
                        break
            if not found:
                steps = 1
                d_idx = 0
            move_type = d_idx * 7 + (steps - 1)
        return from_sq * 73 + move_type

    def alphazero_index_to_move(index, board):
        """
        Kääntää indeksin takaisin UCI–muotoiseksi siirroksi.
        Tämä on yksinkertaistettu toteutus, joka ei kata kaikkia erikoistapauksia.
        """
        from_sq = index // 73
        move_type = index % 73
        if move_type < 56:
            d_idx = move_type // 7
            steps = (move_type % 7) + 1
            directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
            dr, dc = directions[d_idx]
            from_rank = chess.square_rank(from_sq)
            from_file = chess.square_file(from_sq)
            to_rank = from_rank + dr * steps
            to_file = from_file + dc * steps
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = chess.square(to_file, to_rank)
                move = chess.Move(from_sq, to_sq)
            else:
                move = None
        elif move_type < 64:
            knight_index = move_type - 56
            knight_offsets = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
            offset = knight_offsets[knight_index]
            from_rank = chess.square_rank(from_sq)
            from_file = chess.square_file(from_sq)
            to_rank = from_rank + offset[0]
            to_file = from_file + offset[1]
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = chess.square(to_file, to_rank)
                move = chess.Move(from_sq, to_sq)
            else:
                move = None
        else:
            promotion_index = move_type - 64
            from_rank = chess.square_rank(from_sq)
            from_file = chess.square_file(from_sq)
            if from_rank == 6:
                to_rank = 7
            elif from_rank == 1:
                to_rank = 0
            else:
                to_rank = from_rank + 1
            to_sq = chess.square(from_file, to_rank)
            promotion_order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            prom = promotion_order[promotion_index % 4]
            move = chess.Move(from_sq, to_sq, promotion=prom)
        return move

    def get_legal_move_indices(board):
        """Palauttaa listan laillisten siirtojen indekseistä käyttäen AlphaZero–siirtokuvausta."""
        legal_indices = []
        for move in board.legal_moves:
            try:
                idx = get_alphazero_move_index(move, board)
                legal_indices.append(idx)
            except Exception:
                continue
        return legal_indices

#############################################
# 2. Board-to–Tensor Conversion
#############################################
def board_to_tensor(board):
    """
    Muuntaa chess.Board()–olion 12×8×8–tensoriksi.
    Kanavat: [White Pawn, Knight, Bishop, Rook, Queen, King, Black Pawn, Knight, Bishop, Rook, Queen, King]
    """
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            channel = (piece.piece_type - 1) if piece.color == chess.WHITE else (piece.piece_type - 1 + 6)
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            board_tensor[channel, row, col] = 1.0
    return board_tensor

def get_board_tensor(board):
    tensor = board_to_tensor(board)
    if board.turn == chess.BLACK:
        tensor = np.flip(tensor, axis=(1, 2)).copy()  # Varmistetaan, ettei negatiivisia strideja ole.
    return torch.tensor(tensor).unsqueeze(0)  # Muoto: (1, 12, 8, 8)

#############################################
# 3. Residual Network Architecture
#############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, action_size, num_res_blocks=NUM_RES_BLOCKS):
        super().__init__()
        self.conv_input = nn.Conv2d(12, CHANNELS, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(CHANNELS)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList([ResidualBlock(CHANNELS) for _ in range(num_res_blocks)])
        if USE_ADVANCED_MOVE_MAPPING:
            self.conv_policy = nn.Conv2d(CHANNELS, 73, kernel_size=1)
            self.bn_policy = nn.BatchNorm2d(73)
            self.fc_policy = nn.Linear(73 * 8 * 8, action_size)
        else:
            self.conv_policy = nn.Conv2d(CHANNELS, 2, kernel_size=1)
            self.bn_policy = nn.BatchNorm2d(2)
            self.fc_policy = nn.Linear(2 * 8 * 8, action_size)
        self.conv_value = nn.Conv2d(CHANNELS, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)
    def forward(self, x):
        x = self.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        p = self.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.fc_policy(p)
        v = self.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.fc_value1(v))
        value = torch.tanh(self.fc_value2(v))
        return policy_logits, value

#############################################
# 4. Setup Device, Network, Optimizer, and Stockfish Engine
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ChessResNet(ACTION_SIZE, num_res_blocks=NUM_RES_BLOCKS).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

if not install_stockfish_if_needed():
    logging.error("Stockfish is required. Exiting.")
    sys.exit(1)

try:
    stockfish_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
except Exception as e:
    logging.error("Error initializing Stockfish: %s", e)
    stockfish_engine = None

#############################################
# 5. Global Training Metrics (erilliset supervisoidulle ja self–playille)
#############################################
supervised_metrics = {
    "iteration": 0,
    "total_loss": 0.0,
    "policy_loss": 0.0,
    "value_loss": 0.0,
    "target_value": 0.0,
    "predicted_value": 0.0,
    "game": 0,
    "move": 0,
}

self_play_metrics = {
    "iteration": 0,
    "total_loss": 0.0,
    "policy_loss": 0.0,
    "value_loss": 0.0,
    "target_value": 0.0,
    "predicted_value": 0.0,
    "game": "self-play",
    "move": 0,
}

# Global variable for self–play move count
last_self_play_move_count = 0

#############################################
# 6. Supervised Training Loop using Lichess PGN and Stockfish
#############################################
def train_from_lichess(pgn_path, depth=PGN_DEPTH):
    global supervised_metrics
    writer = SummaryWriter()
    if not os.path.exists(pgn_path):
        logging.error("PGN file '%s' not found!", pgn_path)
        return
    with open(pgn_path) as pgn_file:
        game_number = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                pgn_file.seek(0)
                continue
            game_number += 1
            board = game.board()
            move_number = 0
            for move in game.mainline_moves():
                move_number += 1
                board_tensor = get_board_tensor(board).to(device)
                if USE_ADVANCED_MOVE_MAPPING:
                    try:
                        target_index = get_alphazero_move_index(move, board)
                    except Exception:
                        board.push(move)
                        continue
                else:
                    move_uci = move.uci()
                    if move_uci not in move_to_index:
                        board.push(move)
                        continue
                    target_index = move_to_index[move_uci]
                target_value = 0.0
                if stockfish_engine:
                    try:
                        info = stockfish_engine.analyse(board, chess.engine.Limit(depth=depth))
                        score = info["score"].white()
                        if score.is_mate():
                            target_value = 1.0 if score.mate() > 0 else -1.0
                        else:
                            cp = score.score()
                            target_value = max(min(cp / 1000.0, 1.0), -1.0)
                    except Exception as e:
                        logging.warning("Stockfish evaluation error: %s", e)
                        target_value = 0.0
                net.train()
                policy_logits, predicted_value = net(board_tensor)
                policy_logits = policy_logits.squeeze(0)
                predicted_value = predicted_value.squeeze(0)
                target_index_tensor = torch.tensor([target_index], dtype=torch.long, device=device)
                policy_loss = nn.CrossEntropyLoss()(policy_logits.unsqueeze(0), target_index_tensor)
                target_value_tensor = torch.tensor([target_value], dtype=torch.float32, device=device)
                value_loss = nn.MSELoss()(predicted_value, target_value_tensor)
                total_loss = policy_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                iteration = supervised_metrics.get("iteration", 0) + 1
                writer.add_scalar("Loss/Total", total_loss.item(), iteration)
                writer.add_scalar("Loss/Policy", policy_loss.item(), iteration)
                writer.add_scalar("Loss/Value", value_loss.item(), iteration)
                writer.add_scalar("Target/Value", target_value, iteration)

                supervised_metrics = {
                    "iteration": iteration,
                    "total_loss": total_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "target_value": target_value,
                    "predicted_value": predicted_value.item(),
                    "game": game_number,
                    "move": move_number,
                }
                if iteration % 50 == 0:
                    logging.info("[Supervised] Game %d Move %d | Iter %d | Loss: %.4f | Target: %.4f | Predicted: %.4f",
                                 game_number, move_number, iteration, total_loss.item(), target_value, predicted_value.item())
                board.push(move)
    writer.close()

#############################################
# 7. Self–Play (Itsetarkastelu) & Reinforcement Learning
#############################################
replay_buffer = deque(maxlen=10000)
replay_lock = threading.Lock()

def mcts_move(board, num_simulations=50):
    """
    Yksinkertaistettu MCTS–tyyppinen funktio:
      Käy läpi kaikki lailliset siirrot, arvioi verkolla seuraavan tilan ja palauttaa parhaan siirron.
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

def self_play_game():
    """
    Simuloi peliä itsensä vastaan käyttäen MCTS–valintaa.
    Tallentaa jokaisen siirron replay-puskuriin ja palauttaa pelin siirtomäärän.
    """
    global last_self_play_move_count
    states = []
    moves_chosen = []
    board = chess.Board()
    while not board.is_game_over():
        board_tensor = get_board_tensor(board).to(device)
        states.append(board_tensor)
        if random.random() < 0.1:
            move = random.choice(list(board.legal_moves))
        else:
            move = mcts_move(board)
        if USE_ADVANCED_MOVE_MAPPING:
            try:
                move_index = get_alphazero_move_index(move, board)
            except Exception:
                move_index = None
        else:
            move_index = move_to_index.get(move.uci(), None)
        moves_chosen.append(move_index)
        board.push(move)
    result = board.result()
    last_self_play_move_count = len(moves_chosen)
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0
    with replay_lock:
        for state, move_idx in zip(states, moves_chosen):
            if move_idx is not None:
                replay_buffer.append((state, move_idx, outcome))
    logging.info("[Self-Play] Game finished: %s (moves: %d, outcome: %.1f)", result, len(moves_chosen), outcome)

def self_play_training(batch_size=SELF_PLAY_BATCH_SIZE):
    """
    Ottaa satunnaisen erän replay-puskurin näytteitä ja suorittaa vahvistusoppimispäivityksen.
    Päivittää self–play–mittarit käyttäen viimeisimmän pelin siirtojen määrää.
    """
    global self_play_metrics, last_self_play_move_count
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
    total_loss.backward()
    optimizer.step()

    iteration = self_play_metrics.get("iteration", 0) + 1
    self_play_metrics = {
        "iteration": iteration,
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "target_value": outcomes_tensor.mean().item(),
        "predicted_value": predicted_values.mean().item(),
        "game": "self-play",
        "move": last_self_play_move_count,
    }

#############################################
# 8. API Setup with FastAPI
#############################################
app = FastAPI(
    title="Chess AI API",
    description="API to interact with the chess AI trained from Lichess PGN, Stockfish evaluations, and self-play reinforcement learning.",
    version="1.0"
)

@app.get("/predict")
def predict_move(fen: str, epsilon: float = 0.0):
    """
    Antaa annetulle FEN–merkkijonolle tekoälyn siirtoehdotuksen ja arvion.
    Parametrit:
      - fen: FEN–merkkijono (pakollinen)
      - epsilon: Tutkimisen aste (valinnainen, oletus 0.0)
    """
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid FEN string.") from e

    net.eval()
    board_tensor = get_board_tensor(board).to(device)
    with torch.no_grad():
        policy_logits, value = net(board_tensor)
    policy_logits = policy_logits.squeeze(0)
    if USE_ADVANCED_MOVE_MAPPING:
        legal_indices = get_legal_move_indices(board)
    else:
        legal_moves = list(board.legal_moves)
        legal_indices = [move_to_index[m.uci()] for m in legal_moves if m.uci() in move_to_index]
    if not legal_indices:
        raise HTTPException(status_code=500, detail="No legal moves available.")
    mask = torch.full((ACTION_SIZE,), -1e9, device=device)
    mask[legal_indices] = 0.0
    masked_logits = policy_logits + mask
    probs = torch.softmax(masked_logits, dim=0)
    chosen_index = (random.choice(legal_indices) if random.random() < epsilon
                    else torch.multinomial(probs, 1).item())
    if USE_ADVANCED_MOVE_MAPPING:
        chosen_move = alphazero_index_to_move(chosen_index, board)
    else:
        chosen_move = chess.Move.from_uci(index_to_move[chosen_index])
    if chosen_move not in board.legal_moves:
        chosen_move = random.choice(list(board.legal_moves))
    return {"move": chosen_move.uci(), "value_estimate": value.item()}

@app.get("/metrics")
def get_metrics():
    """
    Palauttaa kummankin koulutustyypin viimeisimmät mittarit yhteen yhdistettynä.
    """
    return JSONResponse(content={"supervised": supervised_metrics, "self_play": self_play_metrics})

@app.get("/simulate_game")
def simulate_game():
    """
    Simuloi itsensäpelauspelin käyttäen nykyistä verkkoa.
    Palauttaa pelin lopputuloksen ja pelatut siirrot.
    """
    board = chess.Board()
    moves_list_sim = []
    while not board.is_game_over():
        move = mcts_move(board)
        moves_list_sim.append(move.uci())
        board.push(move)
    return {"result": board.result(), "moves": moves_list_sim}

@app.get("/mcts_predict")
def mcts_predict(fen: str):
    """
    Käyttää yksinkertaistettua MCTS–algoritmia siirron valintaan annetulle FEN–merkkijonolle.
    Parametrit:
      - fen: FEN–merkkijono (pakollinen)
    """
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid FEN string.") from e
    move = mcts_move(board)
    return {"move": move.uci()}

# Dashboard HTML, jossa näytetään erikseen Supervisoidut ja Self-Play mittarit.
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Chess AI Training Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">Chess AI Dashboard</a>
      </div>
    </nav>
    <div class="container">
      <!-- API Endpoints Section -->
      <div class="row mb-4">
        <div class="col">
          <h3>API Endpoints</h3>
          <ul>
            <li>
              <strong><a href="/predict?fen=YOUR_FEN_STRING" target="_blank">/predict</a></strong> – Palauttaa siirtoehdotuksen ja arvion.
            </li>
            <li>
              <strong><a href="/metrics" target="_blank">/metrics</a></strong> – Näyttää Supervisoidut ja Self-Play mittarit.
            </li>
            <li>
              <strong><a href="/simulate_game" target="_blank">/simulate_game</a></strong> – Simuloi itsensäpelauspelin.
            </li>
            <li>
              <strong><a href="/mcts_predict?fen=YOUR_FEN_STRING" target="_blank">/mcts_predict</a></strong> – Valitsee siirron MCTS-algoritmilla.
            </li>
            <li>
              <strong><a href="/docs" target="_blank">/docs</a></strong> – Swagger UI.
            </li>
            <li>
              <strong><a href="/redoc" target="_blank">/redoc</a></strong> – ReDoc dokumentaatio.
            </li>
          </ul>
          <p>Muista korvata <code>YOUR_FEN_STRING</code> omalla FEN–merkkijonollasi.</p>
        </div>
      </div>
      <!-- Supervisoidut Training Metrics -->
      <div class="row mb-4">
        <div class="col">
          <h4>Supervised Training Metrics</h4>
          <table class="table table-striped" id="supervisedTable">
            <!-- Supervisoidut mittarit tulevat tänne -->
          </table>
        </div>
      </div>
      <!-- Self-Play Training Metrics -->
      <div class="row mb-4">
        <div class="col">
          <h4>Self-Play Training Metrics</h4>
          <table class="table table-striped" id="selfPlayTable">
            <!-- Self-play mittarit tulevat tänne -->
          </table>
        </div>
      </div>
    </div>
    <!-- Bootstrap 5 JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
      async function fetchMetrics() {
        try {
          const response = await fetch('/metrics');
          const data = await response.json();
          const supTable = document.getElementById('supervisedTable');
          const spTable = document.getElementById('selfPlayTable');
          
          supTable.innerHTML = `
            <tr><th>Iteration</th><td>${data.supervised.iteration}</td></tr>
            <tr><th>Total Loss</th><td>${data.supervised.total_loss.toFixed(4)}</td></tr>
            <tr><th>Policy Loss</th><td>${data.supervised.policy_loss.toFixed(4)}</td></tr>
            <tr><th>Value Loss</th><td>${data.supervised.value_loss.toFixed(4)}</td></tr>
            <tr><th>Target Value</th><td>${data.supervised.target_value.toFixed(4)}</td></tr>
            <tr><th>Predicted Value</th><td>${data.supervised.predicted_value.toFixed(4)}</td></tr>
            <tr><th>Game</th><td>${data.supervised.game}</td></tr>
            <tr><th>Move in Game</th><td>${data.supervised.move}</td></tr>
          `;
          spTable.innerHTML = `
            <tr><th>Iteration</th><td>${data.self_play.iteration}</td></tr>
            <tr><th>Total Loss</th><td>${data.self_play.total_loss.toFixed(4)}</td></tr>
            <tr><th>Policy Loss</th><td>${data.self_play.policy_loss.toFixed(4)}</td></tr>
            <tr><th>Value Loss</th><td>${data.self_play.value_loss.toFixed(4)}</td></tr>
            <tr><th>Target Value</th><td>${data.self_play.target_value.toFixed(4)}</td></tr>
            <tr><th>Predicted Value</th><td>${data.self_play.predicted_value.toFixed(4)}</td></tr>
            <tr><th>Game</th><td>${data.self_play.game}</td></tr>
            <tr><th>Move in Game</th><td>${data.self_play.move}</td></tr>
          `;
        } catch (error) {
          console.error('Error fetching metrics:', error);
        }
      }
      fetchMetrics();
      setInterval(fetchMetrics, 5000);
    </script>
  </body>
</html>
"""

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Palauttaa dashboard-sivun, joka näyttää erikseen Supervisoidut ja Self-Play mittarit sekä API–linkit."""
    return HTMLResponse(content=dashboard_html, status_code=200)

#############################################
# 9. Training Threads
#############################################
def supervised_training_thread():
    pgn_file = download_lichess_database()
    try:
        train_from_lichess(pgn_file, depth=PGN_DEPTH)
    except Exception as e:
        logging.error("Supervised training encountered an error: %s", e)

def self_play_thread():
    while True:
        try:
            self_play_game()
            self_play_training(batch_size=SELF_PLAY_BATCH_SIZE)
        except Exception as e:
            logging.error("Self–play thread error: %s", e)
        time.sleep(1)

def combined_training_thread():
    threads = []
    t1 = threading.Thread(target=supervised_training_thread, daemon=True)
    threads.append(t1)
    t1.start()
    if USE_SELF_PLAY:
        t2 = threading.Thread(target=self_play_thread, daemon=True)
        threads.append(t2)
        t2.start()
    for t in threads:
        t.join()

#############################################
# 10. Main: Start Training Threads & API Server
#############################################
if __name__ == "__main__":
    trainer_thread = threading.Thread(target=combined_training_thread, daemon=True)
    trainer_thread.start()
    logging.info("Training threads started.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
