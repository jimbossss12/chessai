#!/usr/bin/env python3
"""
Chess AI with Automated Installation of Stockfish and the Lichess October 2020 Database,
Supervised Training from PGN, and a Modern REST API Dashboard.

Features:
  - Checks for Stockfish and installs it using apt-get if missing.
  - Downloads the Lichess PGN database (.zst format) with a progress bar.
  - Decompresses the PGN file with a progress bar.
  - Trains a residual-network–based chess AI via imitation learning using Stockfish evaluations.
  - Serves a modern REST API and a Bootstrap–based dashboard to monitor training.

Dependencies:
    pip install python-chess torch tensorboard fastapi uvicorn requests zstandard tqdm

Run as:
    python chess_ai_supervised_server.py
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

from tqdm import tqdm  # For the progress bars

##############################
# 0. Helper Functions for Installation & Download
##############################
def install_stockfish_if_needed():
    """
    Check if Stockfish is installed. If not, attempt to install it using apt-get.
    Returns True if Stockfish is available.
    """
    stockfish_path = shutil.which("stockfish")
    if stockfish_path:
        print(f"Stockfish found at: {stockfish_path}")
        return True
    else:
        print("Stockfish not found. Attempting to install via apt-get...")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "stockfish"], check=True)
            stockfish_path = shutil.which("stockfish")
            if stockfish_path:
                print(f"Stockfish installed at: {stockfish_path}")
                return True
            else:
                print("Stockfish installation failed.")
                return False
        except Exception as e:
            print("Error installing Stockfish:", e)
            return False

def download_lichess_database():
    """
    Downloads and decompresses the Lichess October 2020 PGN database if not present.
    The compressed file is: lichess_db_standard_rated_2020-10.pgn.zst
    The uncompressed PGN is saved as: lichess_db_standard_rated_2020-10.pgn
    """
    pgn_file = "lichess_db_standard_rated_2020-10.pgn"
    compressed_file = pgn_file + ".zst"
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2020-10.pgn.zst"
    
    if os.path.exists(pgn_file):
        print(f"PGN file '{pgn_file}' already exists.")
        return pgn_file

    # Download the compressed file with a progress bar.
    if not os.path.exists(compressed_file):
        print(f"Downloading Lichess database from {url} ...")
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
            print(f"Downloaded compressed PGN file to {compressed_file}")
        except Exception as e:
            print("Error downloading Lichess database:", e)
            sys.exit(1)

    # Decompress the file using zstandard with a progress bar.
    print("Decompressing the PGN file...")
    try:
        dctx = zstd.ZstdDecompressor()
        compressed_size = os.path.getsize(compressed_file)
        with open(compressed_file, 'rb') as compressed, open(pgn_file, 'wb') as out_file, tqdm(
            total=compressed_size, unit='B', unit_scale=True, desc="Decompressing PGN"
        ) as progress_bar:
            # Use read_to_iter with read_size instead of chunk_size.
            for chunk in dctx.read_to_iter(compressed, read_size=8192):
                out_file.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Decompressed PGN saved as '{pgn_file}'")
    except Exception as e:
        print("Error decompressing PGN file:", e)
        sys.exit(1)
    
    return pgn_file

##############################
# 1. Fixed Move Mapping (Action Space)
##############################
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
print("Action space size:", ACTION_SIZE)

##############################
# 2. Board-to–Tensor Conversion
##############################
def board_to_tensor(board):
    """
    Convert a chess.Board() into a 12x8x8 tensor.
    Channels: [White Pawn, Knight, Bishop, Rook, Queen, King, Black Pawn, Knight, Bishop, Rook, Queen, King]
    """
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type
            channel = piece_type - 1 if piece.color == chess.WHITE else piece_type - 1 + 6
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            board_tensor[channel, row, col] = 1.0
    return board_tensor

def get_board_tensor(board):
    tensor = board_to_tensor(board)
    if board.turn == chess.BLACK:
        tensor = np.flip(tensor, axis=(1, 2))
    return torch.tensor(tensor).unsqueeze(0)  # shape: (1, 12, 8, 8)

##############################
# 3. Residual Network Architecture
##############################
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
    def __init__(self, action_size, num_res_blocks=5):
        super().__init__()
        self.conv_input = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(num_res_blocks)])
        # Policy head.
        self.conv_policy = nn.Conv2d(256, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, action_size)
        # Value head.
        self.conv_value = nn.Conv2d(256, 1, kernel_size=1)
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

##############################
# 4. Setup Device, Network, Optimizer, and Stockfish Engine
##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ChessResNet(ACTION_SIZE, num_res_blocks=5).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

if not install_stockfish_if_needed():
    print("Stockfish is required. Exiting.")
    sys.exit(1)

try:
    stockfish_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
except Exception as e:
    print("Error initializing Stockfish:", e)
    stockfish_engine = None

##############################
# 5. Global Training Metrics
##############################
training_metrics = {
    "iteration": 0,
    "total_loss": 0.0,
    "policy_loss": 0.0,
    "value_loss": 0.0,
    "target_value": 0.0,
    "predicted_value": 0.0,
}

##############################
# 6. Supervised Training Loop using Lichess PGN and Stockfish
##############################
def train_from_lichess(pgn_path, depth=10):
    """
    Reads games from the PGN file and trains the network.
    For each board position:
      - Uses the played move as the target (imitation learning).
      - Uses Stockfish (at the given search depth) to generate a target value.
    """
    global training_metrics
    writer = SummaryWriter()  # Writes logs to ./runs

    if not os.path.exists(pgn_path):
        print(f"PGN file '{pgn_path}' not found!")
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
                            cp = score.score()  # centipawns
                            target_value = max(min(cp / 1000.0, 1.0), -1.0)
                    except Exception as e:
                        print("Stockfish evaluation error:", e)
                        target_value = 0.0

                net.train()
                policy_logits, predicted_value = net(board_tensor)
                policy_logits = policy_logits.squeeze(0)  # (ACTION_SIZE,)
                predicted_value = predicted_value.squeeze(0)

                target_index_tensor = torch.tensor([target_index], dtype=torch.long, device=device)
                policy_loss = nn.CrossEntropyLoss()(policy_logits.unsqueeze(0), target_index_tensor)
                target_value_tensor = torch.tensor([target_value], dtype=torch.float32, device=device)
                value_loss = nn.MSELoss()(predicted_value, target_value_tensor)
                total_loss = policy_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                global_iteration = training_metrics.get("iteration", 0) + 1
                writer.add_scalar("Loss/Total", total_loss.item(), global_iteration)
                writer.add_scalar("Loss/Policy", policy_loss.item(), global_iteration)
                writer.add_scalar("Loss/Value", value_loss.item(), global_iteration)
                writer.add_scalar("Target/Value", target_value, global_iteration)

                training_metrics = {
                    "iteration": global_iteration,
                    "total_loss": total_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "target_value": target_value,
                    "predicted_value": predicted_value.item(),
                    "game": game_number,
                    "move": move_number,
                }

                if global_iteration % 50 == 0:
                    print(f"Game {game_number} Move {move_number} | Iter {global_iteration} | Loss: {total_loss.item():.4f} | Target: {target_value:.4f} | Predicted: {predicted_value.item():.4f}")

                board.push(move)
    writer.close()

def training_thread():
    pgn_file = download_lichess_database()  # Ensure PGN is available.
    try:
        train_from_lichess(pgn_file, depth=10)
    except Exception as e:
        print("Training encountered an error:", e)

##############################
# 7. API Setup with FastAPI
##############################
app = FastAPI(
    title="Chess AI API",
    description="API to interact with the chess AI trained from Lichess PGN and Stockfish.",
    version="1.0"
)

@app.get("/predict")
def predict_move(fen: str, epsilon: float = 0.0):
    """
    Given a FEN string, returns the AI's move prediction and value estimate.
    The optional epsilon (default 0.0) parameter allows exploration.
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
    chosen_move = chess.Move.from_uci(index_to_move[chosen_index])
    if chosen_move not in board.legal_moves:
        chosen_move = random.choice(legal_moves)
    return {"move": chosen_move.uci(), "value_estimate": value.item()}

@app.get("/metrics")
def get_metrics():
    """Return the latest training metrics as JSON."""
    return JSONResponse(content=training_metrics)

# Modern dashboard using Bootstrap 5.
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
      <div class="row mb-4" id="metricsCards">
        <!-- Metrics cards will be inserted here -->
      </div>
      <div class="row">
        <div class="col">
          <h4>Training Metrics</h4>
          <table class="table table-striped">
            <tbody id="metricsTable">
              <!-- Metrics table rows will be inserted here -->
            </tbody>
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
          const cardsContainer = document.getElementById('metricsCards');
          const tableBody = document.getElementById('metricsTable');
          
          // Create cards for key metrics.
          cardsContainer.innerHTML = `
            <div class="col-md-3">
              <div class="card text-white bg-primary mb-3">
                <div class="card-body">
                  <h5 class="card-title">Iteration</h5>
                  <p class="card-text">${data.iteration}</p>
                </div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card text-white bg-success mb-3">
                <div class="card-body">
                  <h5 class="card-title">Total Loss</h5>
                  <p class="card-text">${data.total_loss.toFixed(4)}</p>
                </div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card text-white bg-warning mb-3">
                <div class="card-body">
                  <h5 class="card-title">Policy Loss</h5>
                  <p class="card-text">${data.policy_loss.toFixed(4)}</p>
                </div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card text-white bg-danger mb-3">
                <div class="card-body">
                  <h5 class="card-title">Value Loss</h5>
                  <p class="card-text">${data.value_loss.toFixed(4)}</p>
                </div>
              </div>
            </div>
          `;
          
          // Build table rows for detailed metrics.
          tableBody.innerHTML = `
            <tr><th>Target Value</th><td>${data.target_value.toFixed(4)}</td></tr>
            <tr><th>Predicted Value</th><td>${data.predicted_value.toFixed(4)}</td></tr>
            <tr><th>Game</th><td>${data.game}</td></tr>
            <tr><th>Move in Game</th><td>${data.move}</td></tr>
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
    """Serve the modern dashboard page."""
    return HTMLResponse(content=dashboard_html, status_code=200)

##############################
# 8. Main: Start Training Thread & API Server
##############################
if __name__ == "__main__":
    trainer_thread = threading.Thread(target=training_thread, daemon=True)
    trainer_thread.start()
    print("Training thread started.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
