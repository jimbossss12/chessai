#!/usr/bin/env python3
"""
Chess AI with Automated Installation of Stockfish and Lichess October 2020 Database,
Supervised Training from PGN, and a REST API/Dashboard.

This script will:
  - Check for (and attempt to install) Stockfish via apt-get.
  - Download and decompress the Lichess PGN database (October 2020) from the new .zst URL.
  - Train a residual-network–based chess AI via imitation learning using the PGN.
  - Use Stockfish to provide evaluation targets.
  - Serve a REST API (with FastAPI) for move prediction and training metrics,
    plus a simple HTML dashboard.

Dependencies:
    pip install python-chess torch tensorboard fastapi uvicorn requests zstandard

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

##############################
# 0. Helper Functions for Installation
##############################
def install_stockfish_if_needed():
    """Check if Stockfish is installed. If not, attempt to install it using apt-get."""
    stockfish_path = shutil.which("stockfish")
    if stockfish_path:
        print(f"Stockfish found at: {stockfish_path}")
        return True
    else:
        print("Stockfish not found. Attempting to install via apt-get...")
        try:
            # Update package list and install stockfish.
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
    Download and decompress the Lichess October 2020 PGN database if not present.
    The compressed file is: lichess_db_standard_rated_2020-10.pgn.zst
    The uncompressed PGN will be saved as: lichess_db_standard_rated_2020-10.pgn

    New URL:
      https://database.lichess.org/standard/lichess_db_standard_rated_2020-10.pgn.zst
    """
    pgn_file = "lichess_db_standard_rated_2020-10.pgn"
    compressed_file = pgn_file + ".zst"
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2020-10.pgn.zst"
    
    if os.path.exists(pgn_file):
        print(f"PGN file '{pgn_file}' already exists.")
        return pgn_file

    # If the compressed file is not present, download it.
    if not os.path.exists(compressed_file):
        print(f"Downloading Lichess database from {url} ...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(compressed_file, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)
            print(f"Downloaded compressed PGN file to {compressed_file}")
        except Exception as e:
            print("Error downloading Lichess database:", e)
            sys.exit(1)

    # Decompress the file using zstandard.
    print("Decompressing the PGN file...")
    try:
        dctx = zstd.ZstdDecompressor()
        with open(compressed_file, 'rb') as compressed, open(pgn_file, 'wb') as out_file:
            dctx.copy_stream(compressed, out_file)
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
            # Add promotion moves.
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
        # Policy head.
        p = self.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.fc_policy(p)
        # Value head.
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

# Ensure Stockfish is installed.
if not install_stockfish_if_needed():
    print("Stockfish is required. Exiting.")
    sys.exit(1)

# Initialize Stockfish engine.
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
      - The target move is the move played.
      - Stockfish evaluates the board (using the given search depth) to generate a target value.
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
                # Restart at the beginning of the file.
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

                # Use Stockfish to evaluate the current position.
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
                predicted_value = predicted_value.squeeze(0)  # (1,)

                # Losses.
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
    pgn_file = download_lichess_database()  # Ensure the PGN file is downloaded and decompressed.
    try:
        train_from_lichess(pgn_file, depth=10)
    except Exception as e:
        print("Training encountered an error:", e)

##############################
# 7. API Setup with FastAPI
##############################
app = FastAPI(title="Chess AI API",
              description="API to interact with the chess AI trained from Lichess PGN and Stockfish.",
              version="1.0")

@app.get("/predict")
def predict_move(fen: str, epsilon: float = 0.0):
    """
    Given a FEN string, returns the AI's move prediction and value estimate.
    Optional epsilon (default 0.0) allows exploration.
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

dashboard_html = """
<!DOCTYPE html>
<html>
  <head>
    <title>Chess AI Training Dashboard</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      h1 { color: #333; }
      table { border-collapse: collapse; margin-top: 20px; }
      th, td { border: 1px solid #ccc; padding: 8px 12px; }
      th { background-color: #eee; }
    </style>
  </head>
  <body>
    <h1>Chess AI Training Dashboard</h1>
    <div id="metrics">
      <p>Loading metrics...</p>
    </div>
    <script>
      async function fetchMetrics() {
        try {
          const response = await fetch('/metrics');
          const data = await response.json();
          document.getElementById('metrics').innerHTML = `
            <table>
              <tr><th>Iteration</th><td>${data.iteration}</td></tr>
              <tr><th>Total Loss</th><td>${data.total_loss.toFixed(4)}</td></tr>
              <tr><th>Policy Loss</th><td>${data.policy_loss.toFixed(4)}</td></tr>
              <tr><th>Value Loss</th><td>${data.value_loss.toFixed(4)}</td></tr>
              <tr><th>Target Value</th><td>${data.target_value.toFixed(4)}</td></tr>
              <tr><th>Predicted Value</th><td>${data.predicted_value.toFixed(4)}</td></tr>
              <tr><th>Game</th><td>${data.game}</td></tr>
              <tr><th>Move in Game</th><td>${data.move}</td></tr>
            </table>
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
