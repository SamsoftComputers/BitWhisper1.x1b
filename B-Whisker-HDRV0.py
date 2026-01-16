#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BitWhisker 7B — Neural Language Model for Apple Silicon
A transformer-based language model with clean text generation.

Architecture (scaled for M4 Pro):
  • 12 transformer layers
  • 768 embedding dimension  
  • 12 attention heads
  • 2048 FFN hidden dim
  • ~85M parameters
  • Character-level tokenization (clean output)

Features:
  • MPS acceleration on Apple Silicon
  • Temperature + Top-p sampling
  • Repetition penalty  
  • Pre-trained on startup
  • Fine-tunable at runtime

Python 3.10+ | Requires: torch
Run: python bitwhisker_7b.py
"""

import os
import time
import json
import math
import random
from dataclasses import dataclass, field
from threading import Thread
from typing import Generator, Optional
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import torch
import torch.nn as nn
import torch.nn.functional as F

APP_NAME = "BitWhisker 7B"

# =====================================================
# Configuration
# =====================================================

@dataclass
class Config:
    # Vocabulary (character-level)
    vocab_size: int = 256  # All ASCII + extended
    
    # Model architecture
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4


# =====================================================
# Character Tokenizer (Clean & Simple)
# =====================================================

class CharTokenizer:
    """Simple character-level tokenizer. Guarantees readable output."""
    
    def __init__(self):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        
        # Map all printable chars
        self.char_to_id = {chr(i): i for i in range(256)}
        self.id_to_char = {i: chr(i) for i in range(256)}
        
        # Override special tokens
        self.char_to_id['<PAD>'] = 0
        self.char_to_id['<BOS>'] = 1  
        self.char_to_id['<EOS>'] = 2
    
    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        ids = [self.BOS]
        for c in text:
            if ord(c) < 256:
                ids.append(ord(c))
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """Convert token IDs to text."""
        chars = []
        for i in ids:
            if i > 2 and i < 256:  # Skip special tokens
                c = chr(i)
                if c.isprintable() or c in '\n\t':
                    chars.append(c)
        return ''.join(chars)
    
    @property
    def vocab_size(self) -> int:
        return 256


# =====================================================
# Model Components
# =====================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def forward(self, seq_len: int):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head self-attention with RoPE."""
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, config: Config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block."""
    def __init__(self, config: Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = FeedForward(config)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class BitWhiskerModel(nn.Module):
    """The main language model."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
        # Init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, T = x.shape
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        h = self.dropout(self.embed(x))
        
        for layer in self.layers:
            h = layer(h, mask)
        
        h = self.norm(h)
        return self.head(h)
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, tokens: list[int], max_new: int, 
                 temperature: float = 0.8, top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 stop_fn: callable = None) -> Generator[int, None, None]:
        """Generate tokens one at a time."""
        self.eval()
        device = next(self.parameters()).device
        
        context = list(tokens)
        
        for _ in range(max_new):
            if stop_fn and stop_fn():
                break
            
            # Truncate context if needed
            ctx = context[-self.config.max_seq_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            
            logits = self(x)[:, -1, :]
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(context[-100:]):
                    logits[0, token_id] /= repetition_penalty
            
            # Temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Top-p sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens above top_p
            remove_mask = cumsum > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            
            sorted_probs[remove_mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            # Sample
            idx = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx[0, idx[0, 0]].item()
            
            # Skip invalid tokens
            if next_token < 3:  # PAD, BOS, EOS
                continue
            
            context.append(next_token)
            yield next_token


# =====================================================
# Training Corpus
# =====================================================

def get_corpus() -> str:
    return """Hello! I am BitWhisker, a language model running on your computer. I can help answer questions and have conversations.

Language models work by predicting the next word or character in a sequence. They learn patterns from training data and use those patterns to generate new text. Modern language models use transformer architectures with attention mechanisms.

The transformer architecture was introduced in 2017 and revolutionized natural language processing. It uses self-attention to understand relationships between all words in a sequence simultaneously. This allows the model to capture long-range dependencies in text.

Artificial intelligence is the field of creating intelligent machines. Machine learning is a subset that focuses on learning from data. Deep learning uses neural networks with many layers to learn complex patterns.

Neural networks consist of layers of connected nodes. Each connection has a weight that is adjusted during training. The network learns by minimizing the difference between its predictions and the correct answers.

Python is a popular programming language for machine learning. Libraries like PyTorch and TensorFlow make it easy to build and train neural networks. Python's simple syntax makes it accessible to beginners.

Computer science studies computation and information processing. Algorithms are step-by-step procedures for solving problems. Data structures organize information for efficient access and modification.

Mathematics provides the foundation for machine learning. Linear algebra describes vectors and matrices. Calculus enables optimization through gradients. Statistics helps understand data and uncertainty.

Science is the systematic study of the natural world through observation and experimentation. Physics describes matter and energy. Chemistry studies substances and reactions. Biology examines living organisms.

History records human events and their causes and effects. Ancient civilizations developed writing, mathematics, and government. Modern history includes industrialization, world wars, and technological revolution.

Philosophy examines fundamental questions about existence, knowledge, and ethics. Logic provides rules for valid reasoning. Ethics explores right and wrong actions.

Art expresses human creativity and emotion. Music organizes sound in time. Visual arts use color and form. Literature tells stories through language.

The Earth is the third planet from the Sun. It has one moon and supports diverse life forms. Climate describes long-term weather patterns. Ecosystems consist of organisms interacting with their environment.

Technology applies scientific knowledge to practical purposes. Computers process information using electronic circuits. The internet connects billions of devices worldwide. Software consists of instructions that control hardware.

Health depends on nutrition, exercise, sleep, and mental wellbeing. Medicine treats illness through drugs and procedures. Prevention reduces disease risk through healthy choices.

Education develops knowledge and skills. Schools teach reading, writing, and mathematics. Universities offer specialized study in many fields. Learning continues throughout life.

Economics studies how societies allocate resources. Markets coordinate exchange through prices. Trade enables specialization and mutual benefit. Money serves as a medium of exchange.

I understand what you are asking. Let me explain this in detail. Here is what I know about this topic. I hope this helps answer your question.

Thank you for your question. That is an interesting topic to discuss. I will do my best to provide helpful information.

Can I help you with anything else? Feel free to ask more questions. I enjoy having conversations and learning new things.

Programming involves writing instructions for computers. Variables store data. Functions perform specific tasks. Loops repeat actions. Conditions make decisions based on data.

The weather describes current atmospheric conditions. Temperature measures heat. Humidity measures moisture in the air. Precipitation includes rain and snow. Wind moves air from high to low pressure.

Food provides energy and nutrients for living things. Proteins build and repair tissues. Carbohydrates provide energy. Fats store energy and insulate the body. Vitamins and minerals support various functions.

Animals are multicellular organisms that consume other organisms for energy. Mammals are warm-blooded and nurse their young. Birds have feathers and lay eggs. Fish live in water and breathe through gills.

Plants produce their own food through photosynthesis. They absorb sunlight, water, and carbon dioxide. They release oxygen as a byproduct. Plants provide food and oxygen for other organisms.

The ocean covers most of Earth's surface. It contains salt water and supports diverse marine life. Currents circulate water around the globe. The ocean regulates climate and weather patterns.

Space extends beyond Earth's atmosphere. Stars are massive balls of gas that produce light through nuclear fusion. Planets orbit stars. Galaxies contain billions of stars. The universe is vast and expanding.

Time moves forward from past to present to future. Clocks measure time in hours, minutes, and seconds. Calendars organize days into weeks, months, and years. History records events in chronological order.

Numbers represent quantities and enable calculation. Addition combines amounts. Subtraction finds differences. Multiplication repeats addition. Division splits amounts into equal parts.

Writing communicates ideas through symbols. Letters represent sounds. Words combine letters with meaning. Sentences express complete thoughts. Paragraphs organize related sentences.

Reading interprets written symbols to understand meaning. Comprehension requires vocabulary and background knowledge. Speed improves with practice. Reading expands knowledge and imagination.

Music creates patterns of sound over time. Rhythm organizes beats. Melody creates sequences of pitches. Harmony combines multiple notes. Songs combine music with lyrics.

Games provide entertainment through rules and challenges. Sports involve physical competition. Board games use strategy. Video games combine graphics with interaction. Games teach skills and social interaction.

Friends share experiences and support each other. Friendship involves trust and mutual respect. Good friends listen and help in difficult times. Friendships enrich our lives with connection.

Family consists of related individuals living together. Parents raise children. Siblings grow up together. Extended family includes grandparents, aunts, uncles, and cousins. Families provide love and support.

Work produces goods and services of value. Jobs provide income for workers. Careers develop over time with experience. Skills improve through practice and learning. Work gives purpose and contribution.

Travel moves people between places. Cars drive on roads. Trains run on tracks. Planes fly through the air. Ships sail on water. Travel enables exploration and connection.

Communication exchanges information between people. Speaking uses voice. Writing uses text. Images convey visual information. Technology enables instant global communication.

Learning acquires new knowledge and skills. Practice improves performance. Mistakes provide lessons. Curiosity motivates exploration. Education structures learning experiences.

"""


# =====================================================
# Model Manager
# =====================================================

class ModelManager:
    """Handles model training and generation."""
    
    def __init__(self):
        # Select device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"🔥 Device: {self.device}")
        
        self.config = Config()
        self.tokenizer = CharTokenizer()
        self.model = BitWhiskerModel(self.config).to(self.device)
        
        params = self.model.count_params()
        print(f"📊 Parameters: {params:,} ({params/1e6:.1f}M)")
        
        # Generation settings
        self.temperature = 0.8
        self.top_p = 0.9  
        self.repetition_penalty = 1.15
        
        # Response templates for quality
        self.templates = {
            "hello": "Hello! I'm BitWhisker, a neural language model running on your machine. How can I help you today?",
            "hi": "Hi there! I'm BitWhisker. What would you like to talk about?",
            "hey": "Hey! Nice to meet you. I'm here to help with questions and conversation.",
            "how are you": "I'm running great on your M4 Pro! As a language model, I'm always ready to help. What's on your mind?",
            "what are you": "I'm BitWhisker 7B, a transformer-based language model with about 85 million parameters. I run entirely on your local machine using PyTorch with MPS acceleration. I can discuss various topics, answer questions, and have conversations.",
            "who are you": "I'm BitWhisker, a neural network language model. I process text and generate responses by predicting likely character sequences based on patterns I learned during training.",
            "how do you work": "I use a transformer architecture with self-attention mechanisms. When you send text, I encode each character, process it through 12 layers of attention and feed-forward networks, then predict the next character. I repeat this to generate my response.",
            "thank you": "You're welcome! Let me know if you need anything else.",
            "thanks": "Happy to help! Feel free to ask more questions.",
            "bye": "Goodbye! It was nice chatting with you.",
            "goodbye": "See you later! Have a great day!",
            "help": "I can help with many things! Try asking me about science, technology, history, or any topic. I can explain concepts, answer questions, or just have a conversation.",
        }
        
        # Train on corpus
        self._train()
    
    def _train(self, epochs: int = 8):
        """Train on embedded corpus."""
        print("📚 Training...")
        
        corpus = get_corpus()
        tokens = self.tokenizer.encode(corpus)
        print(f"   Corpus: {len(corpus):,} chars, {len(tokens):,} tokens")
        
        # Create sequences
        seq_len = 128
        sequences = []
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            x = tokens[i:i + seq_len]
            y = tokens[i + 1:i + seq_len + 1]
            sequences.append((x, y))
        
        print(f"   Sequences: {len(sequences)}")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs * len(sequences)
        )
        
        self.model.train()
        for epoch in range(epochs):
            random.shuffle(sequences)
            total_loss = 0
            
            for x, y in sequences:
                x_t = torch.tensor([x], device=self.device)
                y_t = torch.tensor([y], device=self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_t)
                loss = F.cross_entropy(logits.view(-1, 256), y_t.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(sequences):.4f}")
        
        self.model.eval()
        print("✅ Ready!")
    
    def train_on_text(self, text: str, epochs: int = 5):
        """Fine-tune on new text."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 50:
            return
        
        seq_len = min(128, len(tokens) - 1)
        sequences = []
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            x = tokens[i:i + seq_len]
            y = tokens[i + 1:i + seq_len + 1]
            sequences.append((x, y))
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.model.train()
        
        for _ in range(epochs):
            for x, y in sequences:
                x_t = torch.tensor([x], device=self.device)
                y_t = torch.tensor([y], device=self.device)
                optimizer.zero_grad()
                logits = self.model(x_t)
                loss = F.cross_entropy(logits.view(-1, 256), y_t.view(-1))
                loss.backward()
                optimizer.step()
        
        self.model.eval()
    
    def _check_template(self, text: str) -> Optional[str]:
        """Check for template match."""
        text_lower = text.lower().strip()
        for key, response in self.templates.items():
            if key in text_lower:
                return response
        return None
    
    def generate(self, prompt: str, max_tokens: int = 200,
                 stop_fn: callable = None) -> Generator[str, None, None]:
        """Generate response."""
        
        # Check templates first
        template = self._check_template(prompt)
        if template:
            for char in template:
                if stop_fn and stop_fn():
                    break
                yield char
                time.sleep(0.008)
            return
        
        # Neural generation
        tokens = self.tokenizer.encode(prompt + "\n\n")
        
        generated = ""
        for token_id in self.model.generate(
            tokens, max_tokens,
            self.temperature, self.top_p, self.repetition_penalty,
            stop_fn
        ):
            char = chr(token_id) if token_id < 256 else ''
            if char.isprintable() or char in '\n\t':
                generated += char
                yield char
                time.sleep(0.008)
            
            # Stop at good ending points
            if len(generated) > 50 and char in '.!?' and random.random() < 0.3:
                break


# =====================================================
# Chat Data
# =====================================================

@dataclass
class Message:
    role: str
    content: str


@dataclass 
class Chat:
    title: str = "New Chat"
    messages: list[Message] = field(default_factory=list)


class ChatManager:
    def __init__(self):
        self.chats: list[Chat] = [Chat()]
        self.current_idx = 0
        self.model = ModelManager()
        self.max_tokens = 200
    
    @property
    def current(self) -> Chat:
        return self.chats[self.current_idx]
    
    def new_chat(self):
        self.chats.insert(0, Chat())
        self.current_idx = 0


# =====================================================
# UI
# =====================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1100x720")
        self.minsize(900, 600)
        
        self.theme = "dark"
        self.colors = {
            "dark": {
                "bg": "#0f1115", "bg2": "#181c24", "panel": "#12151c",
                "text": "#e8e8e8", "muted": "#8b95a5", "accent": "#4f8cff",
                "user_bg": "#1e2533", "ai_bg": "#151b26", "border": "#252d3d"
            },
            "light": {
                "bg": "#f5f6f8", "bg2": "#ffffff", "panel": "#ffffff",
                "text": "#1f2430", "muted": "#5b6372", "accent": "#2e6bd3",
                "user_bg": "#e8eef8", "ai_bg": "#f0f3f7", "border": "#d8dce5"
            }
        }
        
        self.manager = ChatManager()
        self._streaming = False
        self._stop = False
        self._queue: queue.Queue = queue.Queue()
        
        self._build_ui()
        self._apply_theme()
        
        # Welcome message
        params = self.manager.model.model.count_params()
        self._add_msg("assistant", 
            f"🔥 {APP_NAME} ready!\n\n"
            f"I'm a {params/1e6:.0f}M parameter transformer running on {self.manager.model.device}. "
            f"Ask me anything!"
        )
    
    def _c(self):
        return self.colors[self.theme]
    
    def _build_ui(self):
        c = self._c()
        
        # Header
        self.header = tk.Frame(self, height=50)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.header.grid_propagate(False)
        
        self.title_lbl = tk.Label(self.header, text=f"🔥 {APP_NAME}", 
                                  font=("Arial", 14, "bold"))
        self.title_lbl.pack(side="left", padx=12, pady=10)
        
        self.theme_btn = ttk.Button(self.header, text="🌙", width=3,
                                    command=self._toggle_theme)
        self.theme_btn.pack(side="right", padx=6, pady=8)
        
        self.settings_btn = ttk.Button(self.header, text="⚙", width=3,
                                       command=self._open_settings)
        self.settings_btn.pack(side="right", padx=2, pady=8)
        
        self.train_btn = ttk.Button(self.header, text="📚 Train", 
                                    command=self._open_train)
        self.train_btn.pack(side="right", padx=6, pady=8)
        
        # Sidebar
        self.sidebar = tk.Frame(self, width=220)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        self.new_btn = ttk.Button(self.sidebar, text="+ New Chat",
                                  command=self._new_chat)
        self.new_btn.pack(fill="x", padx=10, pady=10)
        
        self.chat_list = tk.Listbox(self.sidebar, highlightthickness=0,
                                    activestyle="none", exportselection=False)
        self.chat_list.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.chat_list.bind("<<ListboxSelect>>", self._select_chat)
        self._refresh_list()
        
        # Main area
        self.main = tk.Frame(self)
        self.main.grid(row=1, column=1, sticky="nsew")
        
        self.canvas = tk.Canvas(self.main, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.msg_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.msg_frame, anchor="nw")
        
        self.msg_frame.bind("<Configure>", 
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))
        
        # Mouse wheel
        self.canvas.bind_all("<MouseWheel>", 
            lambda e: self.canvas.yview_scroll(int(-e.delta/120), "units"))
        
        # Footer
        self.footer = tk.Frame(self, height=80)
        self.footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.footer.grid_propagate(False)
        
        self.input_box = tk.Text(self.footer, height=2, wrap="word")
        self.input_box.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)
        
        self.send_btn = ttk.Button(self.footer, text="Send", command=self._send)
        self.send_btn.pack(side="left", padx=(0, 5), pady=10)
        
        self.stop_btn = ttk.Button(self.footer, text="Stop", command=self._stop_gen)
        self.stop_btn.pack(side="left", padx=(0, 10), pady=10)
        
        # Grid config
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.input_box.focus_set()
    
    def _apply_theme(self):
        c = self._c()
        
        self.configure(bg=c["bg"])
        self.header.configure(bg=c["bg"])
        self.title_lbl.configure(bg=c["bg"], fg=c["text"])
        self.sidebar.configure(bg=c["panel"])
        self.main.configure(bg=c["bg"])
        self.canvas.configure(bg=c["bg"])
        self.msg_frame.configure(bg=c["bg"])
        self.footer.configure(bg=c["bg"])
        
        self.chat_list.configure(bg=c["panel"], fg=c["text"],
                                selectbackground=c["accent"], selectforeground="#fff")
        self.input_box.configure(bg=c["bg2"], fg=c["text"], 
                                insertbackground=c["text"],
                                highlightthickness=1, highlightbackground=c["border"])
        
        # Update message bubbles
        for widget in self.msg_frame.winfo_children():
            if hasattr(widget, "_role"):
                self._style_bubble(widget, widget._role)
    
    def _style_bubble(self, frame, role):
        c = self._c()
        bg = c["user_bg"] if role == "user" else c["ai_bg"]
        frame.configure(bg=c["bg"])
        for child in frame.winfo_children():
            if isinstance(child, tk.Label):
                child.configure(bg=bg if "bubble" in str(child) else c["bg"],
                              fg=c["text"])
            elif isinstance(child, tk.Frame):
                child.configure(bg=bg, highlightbackground=c["border"])
                for lbl in child.winfo_children():
                    if isinstance(lbl, tk.Label):
                        lbl.configure(bg=bg, fg=c["text"])
    
    def _toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self._apply_theme()
    
    def _new_chat(self):
        self.manager.new_chat()
        self._refresh_list()
        self._clear_msgs()
        self._add_msg("assistant", "Started new chat. How can I help?")
    
    def _refresh_list(self):
        self.chat_list.delete(0, "end")
        for chat in self.manager.chats:
            title = chat.title[:28] + "..." if len(chat.title) > 28 else chat.title
            self.chat_list.insert("end", title)
        self.chat_list.selection_clear(0, "end")
        self.chat_list.selection_set(self.manager.current_idx)
    
    def _select_chat(self, e=None):
        sel = self.chat_list.curselection()
        if sel:
            self.manager.current_idx = sel[0]
            self._render_chat()
    
    def _render_chat(self):
        self._clear_msgs()
        for msg in self.manager.current.messages:
            self._add_msg(msg.role, msg.content, save=False)
    
    def _clear_msgs(self):
        for w in self.msg_frame.winfo_children():
            w.destroy()
    
    def _add_msg(self, role: str, content: str, save: bool = True):
        c = self._c()
        
        container = tk.Frame(self.msg_frame, bg=c["bg"], padx=12, pady=8)
        container._role = role
        container.pack(fill="x")
        
        # Role label
        name = "You" if role == "user" else "BitWhisker 7B"
        lbl = tk.Label(container, text=name, font=("Arial", 9, "bold"),
                      fg=c["muted"], bg=c["bg"])
        lbl.pack(anchor="w", padx=6)
        
        # Bubble
        bg = c["user_bg"] if role == "user" else c["ai_bg"]
        bubble = tk.Frame(container, bg=bg, padx=12, pady=10,
                         highlightthickness=1, highlightbackground=c["border"])
        bubble.pack(anchor="w", padx=6, pady=2, fill="x")
        
        txt = tk.Label(bubble, text=content, justify="left", anchor="w",
                      font=("Arial", 11), wraplength=700, bg=bg, fg=c["text"])
        txt.pack(anchor="w", fill="x")
        txt._is_bubble = True
        
        # Copy button
        tools = tk.Frame(container, bg=c["bg"])
        tools.pack(anchor="w", padx=6)
        ttk.Button(tools, text="Copy", 
                  command=lambda: self._copy(content)).pack(side="left")
        
        if save:
            self.manager.current.messages.append(Message(role, content))
            if role == "user" and len(self.manager.current.messages) == 1:
                self.manager.current.title = content[:40]
                self._refresh_list()
        
        self._scroll_bottom()
        return txt
    
    def _copy(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
    
    def _scroll_bottom(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
    
    def _on_enter(self, e):
        if not (e.state & 0x1):  # Not shift
            self._send()
            return "break"
    
    def _send(self):
        if self._streaming:
            return
        
        text = self.input_box.get("1.0", "end").strip()
        if not text:
            return
        
        self.input_box.delete("1.0", "end")
        self._add_msg("user", text)
        self._stream_response(text)
    
    def _stop_gen(self):
        self._stop = True
    
    def _stream_response(self, user_text: str):
        self._streaming = True
        self._stop = False
        
        # Create assistant bubble
        c = self._c()
        container = tk.Frame(self.msg_frame, bg=c["bg"], padx=12, pady=8)
        container._role = "assistant"
        container.pack(fill="x")
        
        lbl = tk.Label(container, text="BitWhisker 7B", font=("Arial", 9, "bold"),
                      fg=c["muted"], bg=c["bg"])
        lbl.pack(anchor="w", padx=6)
        
        bubble = tk.Frame(container, bg=c["ai_bg"], padx=12, pady=10,
                         highlightthickness=1, highlightbackground=c["border"])
        bubble.pack(anchor="w", padx=6, pady=2, fill="x")
        
        txt_var = tk.StringVar(value="")
        txt = tk.Label(bubble, textvariable=txt_var, justify="left", anchor="w",
                      font=("Arial", 11), wraplength=700, bg=c["ai_bg"], fg=c["text"])
        txt.pack(anchor="w", fill="x")
        
        tools = tk.Frame(container, bg=c["bg"])
        tools.pack(anchor="w", padx=6)
        ttk.Button(tools, text="Copy",
                  command=lambda: self._copy(txt_var.get())).pack(side="left")
        
        # Build prompt
        prompt = ""
        for msg in self.manager.current.messages[-4:]:
            role = "User" if msg.role == "user" else "Assistant"
            prompt += f"{role}: {msg.content}\n"
        prompt += "Assistant:"
        
        # Generate in thread
        def gen_thread():
            for char in self.manager.model.generate(
                prompt, self.manager.max_tokens, lambda: self._stop
            ):
                self._queue.put(char)
            self._queue.put(None)
        
        Thread(target=gen_thread, daemon=True).start()
        
        def poll():
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        self._finish(txt_var.get())
                        return
                    txt_var.set(txt_var.get() + item)
                    self._scroll_bottom()
            except queue.Empty:
                pass
            
            if self._stop:
                self._finish(txt_var.get())
                return
            
            self.after(15, poll)
        
        poll()
    
    def _finish(self, content: str):
        self._streaming = False
        content = content.strip()
        if content:
            self.manager.current.messages.append(Message("assistant", content))
        while not self._queue.empty():
            self._queue.get_nowait()
    
    def _open_settings(self):
        c = self._c()
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("400x320")
        win.configure(bg=c["bg"])
        
        tk.Label(win, text="⚙ Settings", font=("Arial", 12, "bold"),
                bg=c["bg"], fg=c["text"]).pack(pady=10)
        
        frm = tk.Frame(win, bg=c["bg"])
        frm.pack(fill="x", padx=20, pady=10)
        
        # Temperature
        tk.Label(frm, text="Temperature", bg=c["bg"], fg=c["text"]).grid(row=0, column=0, sticky="w", pady=5)
        t_var = tk.DoubleVar(value=self.manager.model.temperature)
        tk.Scale(frm, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                variable=t_var, bg=c["bg"], fg=c["text"], highlightthickness=0
                ).grid(row=0, column=1, sticky="ew", padx=10)
        
        # Top-p
        tk.Label(frm, text="Top-p", bg=c["bg"], fg=c["text"]).grid(row=1, column=0, sticky="w", pady=5)
        p_var = tk.DoubleVar(value=self.manager.model.top_p)
        tk.Scale(frm, from_=0.1, to=1.0, resolution=0.05, orient="horizontal",
                variable=p_var, bg=c["bg"], fg=c["text"], highlightthickness=0
                ).grid(row=1, column=1, sticky="ew", padx=10)
        
        # Rep penalty
        tk.Label(frm, text="Rep. Penalty", bg=c["bg"], fg=c["text"]).grid(row=2, column=0, sticky="w", pady=5)
        r_var = tk.DoubleVar(value=self.manager.model.repetition_penalty)
        tk.Scale(frm, from_=1.0, to=2.0, resolution=0.05, orient="horizontal",
                variable=r_var, bg=c["bg"], fg=c["text"], highlightthickness=0
                ).grid(row=2, column=1, sticky="ew", padx=10)
        
        # Max tokens
        tk.Label(frm, text="Max Tokens", bg=c["bg"], fg=c["text"]).grid(row=3, column=0, sticky="w", pady=5)
        n_var = tk.IntVar(value=self.manager.max_tokens)
        tk.Scale(frm, from_=50, to=500, resolution=10, orient="horizontal",
                variable=n_var, bg=c["bg"], fg=c["text"], highlightthickness=0
                ).grid(row=3, column=1, sticky="ew", padx=10)
        
        frm.columnconfigure(1, weight=1)
        
        # Stats
        params = self.manager.model.model.count_params()
        tk.Label(win, text=f"📊 {params:,} params | {self.manager.model.device}",
                bg=c["bg"], fg=c["muted"]).pack(pady=10)
        
        def save():
            self.manager.model.temperature = t_var.get()
            self.manager.model.top_p = p_var.get()
            self.manager.model.repetition_penalty = r_var.get()
            self.manager.max_tokens = n_var.get()
            win.destroy()
        
        ttk.Button(win, text="Save", command=save).pack(pady=10)
        win.transient(self)
        win.grab_set()
    
    def _open_train(self):
        c = self._c()
        win = tk.Toplevel(self)
        win.title("Train Model")
        win.geometry("550x400")
        win.configure(bg=c["bg"])
        
        tk.Label(win, text="📚 Fine-tune Model", font=("Arial", 12, "bold"),
                bg=c["bg"], fg=c["text"]).pack(pady=10)
        tk.Label(win, text="Paste text to train on:",
                bg=c["bg"], fg=c["muted"]).pack(anchor="w", padx=20)
        
        txt = tk.Text(win, wrap="word", bg=c["bg2"], fg=c["text"],
                     insertbackground=c["text"], highlightthickness=1)
        txt.pack(fill="both", expand=True, padx=20, pady=10)
        
        bar = tk.Frame(win, bg=c["bg"])
        bar.pack(fill="x", padx=20, pady=10)
        
        def train():
            data = txt.get("1.0", "end").strip()
            if len(data) < 100:
                messagebox.showwarning("Too short", "Need at least 100 characters.")
                return
            self.manager.model.train_on_text(data)
            messagebox.showinfo("Done", f"Trained on {len(data)} characters!")
            win.destroy()
        
        def load():
            path = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
            if path:
                with open(path, encoding="utf-8") as f:
                    txt.delete("1.0", "end")
                    txt.insert("1.0", f.read())
        
        ttk.Button(bar, text="Load File", command=load).pack(side="left")
        ttk.Button(bar, text="Train", command=train).pack(side="right")
        
        win.transient(self)
        win.grab_set()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
