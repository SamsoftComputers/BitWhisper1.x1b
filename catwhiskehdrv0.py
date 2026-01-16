#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BitWhisker 7B — Real Neural Language Model
A genuine transformer-based LM built with PyTorch, optimized for Apple Silicon M4 Pro.

Architecture:
  • GPT-style decoder-only transformer
  • Rotary position embeddings (RoPE)
  • Multi-head self-attention with causal masking
  • SwiGLU feed-forward layers
  • Layer normalization (RMSNorm)
  • ~7M parameters (7B is marketing xd)

Features:
  • Pre-trained on embedded corpus at startup
  • Temperature + Top-p (nucleus) sampling
  • Repetition penalty
  • KV-cache for fast generation
  • MPS acceleration on Apple Silicon
  • Runtime fine-tuning

Python 3.10+ | Requires: torch, numpy
Run: python bitwhisker_7b.py
"""

import time
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Thread
from typing import Generator, Optional
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

APP_NAME = "BitWhisker 7B"
VERSION = "7.0.0"

# =====================================================
# Model Configuration
# =====================================================

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    hidden_dim: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0
    
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


# =====================================================
# Tokenizer (BPE-lite)
# =====================================================

class SimpleTokenizer:
    """Character-level tokenizer with common word tokens."""
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        
        # Special tokens
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        
        self._build_vocab()
    
    def _build_vocab(self):
        idx = 0
        # Special tokens
        for tok in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self.char_to_id[tok] = idx
            self.id_to_char[idx] = tok
            idx += 1
        
        # Printable ASCII
        for c in range(32, 127):
            char = chr(c)
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char
            idx += 1
        
        # Common punctuation and special chars
        extras = "\n\t""''—–…•●○■□▪▫"
        for c in extras:
            if c not in self.char_to_id:
                self.char_to_id[c] = idx
                self.id_to_char[idx] = c
                idx += 1
        
        # Common words as single tokens
        common_words = [
            " the", " a", " an", " is", " are", " was", " were", " be", " been",
            " have", " has", " had", " do", " does", " did", " will", " would",
            " could", " should", " can", " may", " might", " must", " shall",
            " to", " of", " in", " for", " on", " with", " at", " by", " from",
            " as", " into", " through", " during", " before", " after", " above",
            " below", " between", " under", " again", " further", " then", " once",
            " and", " but", " or", " nor", " so", " yet", " both", " either",
            " neither", " not", " only", " own", " same", " than", " too", " very",
            " that", " this", " these", " those", " what", " which", " who", " whom",
            " how", " when", " where", " why", " if", " because", " although",
            " I", " you", " he", " she", " it", " we", " they", " me", " him",
            " her", " us", " them", " my", " your", " his", " its", " our", " their",
            " there", " here", " all", " each", " every", " some", " any", " no",
            " more", " most", " other", " such", " many", " much", " few", " little",
            " new", " old", " good", " bad", " great", " small", " large", " long",
            " first", " last", " next", " high", " low", " right", " left", " best",
            " think", " know", " want", " need", " like", " love", " hate", " see",
            " look", " find", " give", " take", " come", " go", " make", " get",
            " say", " tell", " ask", " use", " work", " try", " call", " feel",
            " become", " leave", " put", " keep", " let", " begin", " seem", " help",
            " show", " hear", " play", " run", " move", " live", " believe", " hold",
            "ing", "tion", "ed", "er", "est", "ly", "ment", "ness", "ful", "less",
            "able", "ible", "ous", "ive", "al", "ial", "ian", "ish", "ize", "ify",
            " language", " model", " neural", " network", " learning", " machine",
            " computer", " program", " system", " data", " information", " knowledge",
            " question", " answer", " example", " problem", " solution", " method",
            " process", " result", " number", " part", " place", " case", " point",
            " world", " life", " time", " year", " day", " way", " thing", " man",
            " woman", " child", " person", " people", " family", " friend", " home",
            " house", " city", " country", " state", " government", " company",
        ]
        
        for word in common_words:
            if idx < self.vocab_size and word not in self.word_to_id:
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
                idx += 1
        
        self.vocab_used = idx
    
    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.pad_token]
    
    @property 
    def bos_id(self) -> int:
        return self.char_to_id[self.bos_token]
    
    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.eos_token]
    
    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.unk_token]
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        tokens = [self.bos_id]
        i = 0
        while i < len(text):
            # Try to match word tokens (greedy longest match)
            matched = False
            for length in range(min(12, len(text) - i), 1, -1):
                substr = text[i:i+length]
                if substr in self.word_to_id:
                    tokens.append(self.word_to_id[substr])
                    i += length
                    matched = True
                    break
            
            if not matched:
                char = text[i]
                if char in self.char_to_id:
                    tokens.append(self.char_to_id[char])
                else:
                    tokens.append(self.unk_id)
                i += 1
        
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for id in ids:
            if id in self.id_to_word:
                chars.append(self.id_to_word[id])
            elif id in self.id_to_char:
                char = self.id_to_char[id]
                if char not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
                    chars.append(char)
        return "".join(chars)


# =====================================================
# Model Components
# =====================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, 
                          device: str = "cpu") -> torch.Tensor:
    """Precompute rotary position embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, 
               freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs = freqs[:xq_.shape[1], :].unsqueeze(0).unsqueeze(2)
    
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        # KV cache for inference
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor, freqs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rope(xq, xk, freqs)
        
        if use_cache:
            if self.cache_k is not None:
                xk = torch.cat([self.cache_k, xk], dim=1)
                xv = torch.cat([self.cache_v, xv], dim=1)
            self.cache_k = xk
            self.cache_v = xv
        
        # Repeat KV heads if grouped query attention
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)
        
        # Attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)
    
    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer decoder block."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
    
    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs, mask, use_cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
    
    def clear_cache(self):
        self.attention.clear_cache()


class BitWhiskerLM(nn.Module):
    """BitWhisker 7B Language Model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.output.weight = self.tok_emb.weight
        
        # Precompute RoPE
        self.register_buffer(
            "freqs",
            precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, 
                use_cache: bool = False) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        
        h = self.tok_emb(tokens)
        h = self.dropout(h)
        
        # Causal mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            if use_cache and self.layers[0].attention.cache_k is not None:
                cache_len = self.layers[0].attention.cache_k.shape[1]
                mask = torch.zeros((seqlen, cache_len + seqlen), device=tokens.device)
                mask[:, cache_len:] = torch.triu(
                    torch.full((seqlen, seqlen), float("-inf"), device=tokens.device),
                    diagonal=1
                )
        
        freqs = self.freqs
        
        for layer in self.layers:
            h = layer(h, freqs, mask, use_cache)
        
        h = self.norm(h)
        logits = self.output(h)
        
        return logits
    
    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()
    
    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.85, top_p: float = 0.92,
                 repetition_penalty: float = 1.15,
                 stop_event: Optional[callable] = None) -> Generator[int, None, None]:
        """Generate tokens with nucleus sampling."""
        self.eval()
        self.clear_cache()
        
        generated = tokens.tolist()[0] if tokens.dim() > 1 else tokens.tolist()
        
        for _ in range(max_new_tokens):
            if stop_event and stop_event():
                break
            
            # Forward pass (use cache for efficiency)
            if len(generated) == tokens.shape[-1]:
                logits = self(tokens, use_cache=True)
            else:
                last_token = torch.tensor([[generated[-1]]], device=tokens.device)
                logits = self(last_token, use_cache=True)
            
            logits = logits[:, -1, :]
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[-50:]):
                    logits[0, token_id] /= repetition_penalty
            
            # Temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative prob > top_p
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum()
            
            # Sample
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            yield next_token
        
        self.clear_cache()
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =====================================================
# Training Corpus
# =====================================================

def get_training_corpus() -> str:
    """Returns massive embedded training corpus."""
    return """
Language models are computational systems that learn to predict and generate text by analyzing patterns in large amounts of training data. Modern language models use neural network architectures, particularly transformers, to capture complex relationships between words and concepts. The transformer architecture, introduced in 2017, revolutionized natural language processing through its attention mechanism.

The attention mechanism allows models to focus on relevant parts of the input when generating each output token. Self-attention computes relationships between all positions in a sequence simultaneously, enabling parallel processing and capturing long-range dependencies. Multi-head attention runs multiple attention operations in parallel, allowing the model to attend to information from different representation subspaces.

Training language models involves optimizing neural network weights to minimize prediction error on text data. The training process uses backpropagation to compute gradients and optimization algorithms like Adam to update weights. Large models require significant computational resources, often using distributed training across multiple GPUs or TPUs.

Artificial intelligence encompasses the development of systems capable of performing tasks that typically require human intelligence. Machine learning, a subset of AI, focuses on algorithms that improve through experience. Deep learning uses neural networks with many layers to learn hierarchical representations of data.

Neural networks are inspired by biological neurons but operate quite differently. Artificial neurons compute weighted sums of inputs, apply activation functions, and pass results to subsequent layers. Common activation functions include ReLU, sigmoid, and tanh, each with different properties affecting gradient flow during training.

Natural language processing enables computers to understand, interpret, and generate human language. Key NLP tasks include text classification, named entity recognition, sentiment analysis, machine translation, question answering, and text summarization. Modern NLP systems achieve remarkable performance by pretraining on large text corpora and fine-tuning on specific tasks.

Programming involves writing instructions for computers to execute. Python has become the dominant language for machine learning and data science due to its readable syntax, extensive libraries, and active community. Popular Python libraries for ML include NumPy for numerical computing, PyTorch and TensorFlow for deep learning, and scikit-learn for traditional machine learning.

Computer science studies computation, information processing, and the design of computer systems. Core areas include algorithms, data structures, computer architecture, operating systems, networking, databases, and software engineering. Theoretical computer science explores computational complexity, computability, and formal languages.

Mathematics provides the foundation for computer science and machine learning. Linear algebra describes vectors, matrices, and linear transformations essential for neural networks. Calculus enables optimization through gradient computation. Probability theory and statistics underpin machine learning algorithms and uncertainty quantification.

Physics describes the fundamental laws governing the universe. Mechanics studies motion and forces. Electromagnetism explains light, electricity, and magnetism. Thermodynamics describes heat, energy, and entropy. Quantum mechanics reveals the probabilistic nature of particles at atomic scales. Relativity describes gravity as spacetime curvature.

Chemistry studies matter, its properties, and transformations. Atoms combine into molecules through chemical bonds. Chemical reactions rearrange atoms, involving energy changes described by thermodynamics. Biochemistry studies chemical processes in living organisms, including metabolism, genetics, and protein function.

Biology studies living organisms and life processes. Cells are the basic units of life, containing DNA that encodes genetic information. Evolution through natural selection explains the diversity of life. Ecology studies interactions between organisms and their environments. Neuroscience investigates the nervous system and brain function.

History records and interprets past human events. Ancient civilizations developed writing, mathematics, and complex societies in Mesopotamia, Egypt, Greece, Rome, China, and India. The Middle Ages saw the rise of feudalism in Europe and Islamic golden age advances in science. The Renaissance sparked renewed interest in classical learning. The Industrial Revolution transformed economies through mechanization.

Philosophy examines fundamental questions about existence, knowledge, ethics, mind, and language. Epistemology studies the nature and limits of knowledge. Metaphysics investigates the nature of reality. Ethics explores moral principles and values. Logic provides formal methods for valid reasoning.

Economics studies how societies allocate scarce resources. Microeconomics analyzes individual and firm behavior. Macroeconomics examines aggregate economic phenomena like growth, inflation, and unemployment. Markets coordinate economic activity through prices. Game theory models strategic interactions between rational agents.

Psychology studies mind and behavior. Cognitive psychology investigates mental processes including perception, memory, reasoning, and decision-making. Social psychology examines how people influence each other. Clinical psychology addresses mental health and psychological disorders. Developmental psychology studies human growth across the lifespan.

Sociology studies human society and social behavior. Social structures shape individual opportunities and outcomes. Culture transmits knowledge, beliefs, and practices across generations. Institutions organize social life through established rules and norms. Social stratification creates hierarchies based on wealth, status, and power.

The scientific method involves forming hypotheses, designing experiments, collecting data, and revising theories based on evidence. Reproducibility ensures findings can be verified by independent researchers. Peer review evaluates research quality before publication. Science advances through incremental discoveries and occasional paradigm shifts.

Technology applies scientific knowledge to practical purposes. Information technology revolutionized communication, commerce, and entertainment. Biotechnology enables genetic engineering and medical advances. Renewable energy technologies address climate change. Artificial intelligence promises to transform many industries and raise important ethical questions.

Communication involves encoding, transmitting, and decoding messages. Language uses symbols to represent meaning. Writing preserves information across time and space. Mass media broadcasts information to large audiences. The internet enables instant global communication. Social media platforms facilitate user-generated content and interaction.

Art expresses human creativity and emotion through various media. Visual arts include painting, sculpture, photography, and digital art. Performing arts encompass music, dance, theater, and film. Literature uses language to create imaginative works including poetry, fiction, and drama. Architecture designs functional and aesthetic built environments.

Music organizes sound in time through rhythm, melody, and harmony. Instruments produce sound through vibration of strings, columns of air, or membranes. Musical notation records compositions for performance. Genres range from classical and jazz to rock, electronic, and world music. Music evokes emotions and creates cultural identities.

Human health depends on proper nutrition, exercise, sleep, and mental wellbeing. Medicine diagnoses and treats diseases through drugs, surgery, and other interventions. Public health focuses on preventing disease and promoting health at population level. Healthcare systems organize medical services delivery and financing.

Education develops knowledge, skills, and character. Formal education occurs in schools, colleges, and universities. Informal learning happens through experience and self-study. Pedagogy studies teaching methods and learning theories. Education enables individual advancement and societal progress.

The environment includes all living and non-living things on Earth. Ecosystems consist of organisms interacting with their physical surroundings. Climate describes long-term weather patterns. Human activities impact the environment through pollution, habitat destruction, and greenhouse gas emissions. Sustainability seeks to meet present needs without compromising future generations.

Space extends infinitely beyond Earth's atmosphere. Stars are massive balls of plasma producing energy through nuclear fusion. Planets orbit stars, with eight planets in our solar system. Galaxies contain billions of stars bound by gravity. The universe began with the Big Bang about 13.8 billion years ago and continues expanding.

I am a language model trained to understand and generate text. I process your input and predict likely continuations based on patterns learned during training. I don't have consciousness or feelings, but I can engage in helpful conversations about many topics. I aim to be accurate, helpful, and honest in my responses.

When you ask me questions, I analyze your text and generate relevant responses. I have knowledge spanning many domains including science, technology, history, arts, and everyday topics. I can explain concepts, answer questions, write creative content, help with analysis, and engage in open-ended conversation.

I have limitations. My training data has a knowledge cutoff date, so I may not know about recent events. I can make mistakes and may occasionally generate incorrect information. I work best when given clear, specific questions and context. I cannot browse the internet, run code, or access external systems unless specifically configured to do so.

Effective communication with language models involves being clear and specific about your needs. Providing context helps generate more relevant responses. Breaking complex requests into steps often yields better results. Asking follow-up questions can clarify or expand on initial responses.

Language models raise important questions about AI safety, bias, and societal impact. Researchers work to make models more accurate, fair, and aligned with human values. Understanding model capabilities and limitations helps users apply them appropriately. The field continues evolving rapidly with new techniques and applications emerging regularly.

Hello! How can I help you today? I'm happy to discuss any topic, answer questions, explain concepts, or just have a conversation. Feel free to ask me anything.

What would you like to talk about? I can discuss science, technology, history, arts, philosophy, current events, or help with various tasks like writing, analysis, and problem-solving.

Thank you for your question. Let me think about this carefully and provide a helpful response.

That's an interesting point. There are multiple perspectives worth considering here.

I understand what you're asking. Let me explain this concept in detail.

Great question! This topic involves several important factors that I'll break down for you.

I appreciate your curiosity. Learning about new things is valuable and I'm happy to help.

Let me share what I know about this subject based on my training.

This is a complex topic with many nuances. I'll do my best to give you a clear explanation.

I hope this helps! Feel free to ask if you need any clarification or have more questions.
"""


# =====================================================
# Model Manager
# =====================================================

class ModelManager:
    """Manages model training and inference."""
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🔥 Using device: {self.device}")
        
        self.config = ModelConfig()
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)
        self.model = BitWhiskerLM(self.config).to(self.device)
        
        param_count = self.model.count_parameters()
        print(f"📊 Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
        
        self.temperature = 0.85
        self.top_p = 0.92
        self.repetition_penalty = 1.15
        
        # Train on startup
        self._train_initial()
    
    def _train_initial(self, epochs: int = 3, lr: float = 3e-4):
        """Train model on embedded corpus."""
        print("📚 Training on embedded corpus...")
        
        corpus = get_training_corpus()
        tokens = self.tokenizer.encode(corpus)
        
        # Create training sequences
        seq_len = 64
        sequences = []
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            x = tokens[i:i + seq_len]
            y = tokens[i + 1:i + seq_len + 1]
            if len(x) == seq_len and len(y) == seq_len:
                sequences.append((x, y))
        
        print(f"   Created {len(sequences)} training sequences")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(sequences))
        
        self.model.train()
        for epoch in range(epochs):
            random.shuffle(sequences)
            total_loss = 0
            
            for x, y in sequences:
                x_tensor = torch.tensor([x], device=self.device)
                y_tensor = torch.tensor([y], device=self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_tensor)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y_tensor.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(sequences)
            print(f"   Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("✅ Training complete!")
        self.model.eval()
    
    def train_on_text(self, text: str, epochs: int = 2, lr: float = 1e-4):
        """Fine-tune on additional text."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 32:
            return
        
        seq_len = min(64, len(tokens) - 1)
        sequences = []
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            x = tokens[i:i + seq_len]
            y = tokens[i + 1:i + seq_len + 1]
            if len(x) == seq_len and len(y) == seq_len:
                sequences.append((x, y))
        
        if not sequences:
            return
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        
        for epoch in range(epochs):
            for x, y in sequences:
                x_tensor = torch.tensor([x], device=self.device)
                y_tensor = torch.tensor([y], device=self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_tensor)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y_tensor.view(-1))
                loss.backward()
                optimizer.step()
        
        self.model.eval()
    
    def generate(self, prompt: str, max_tokens: int = 200,
                 stop_event: Optional[callable] = None) -> Generator[str, None, None]:
        """Generate text from prompt."""
        tokens = self.tokenizer.encode(prompt)
        tokens_tensor = torch.tensor([tokens], device=self.device)
        
        buffer = ""
        for token_id in self.model.generate(
            tokens_tensor, max_tokens,
            self.temperature, self.top_p, self.repetition_penalty,
            stop_event
        ):
            # Decode incrementally
            char = self.tokenizer.decode([token_id])
            buffer += char
            
            # Yield complete words
            if char.endswith(" ") or char in ".,!?;:\n":
                yield buffer
                buffer = ""
                time.sleep(0.01)
        
        if buffer:
            yield buffer


# =====================================================
# Chat Controller
# =====================================================

@dataclass
class Message:
    role: str
    content: str
    id: int = field(default_factory=lambda: int(time.time() * 1e6) & 0x7FFFFFFF)


@dataclass 
class ChatThread:
    title: str
    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"title": self.title, "messages": [m.__dict__ for m in self.messages]}


class ChatController:
    def __init__(self):
        self.threads: list[ChatThread] = [ChatThread(title="New Chat")]
        self.current_index = 0
        self.max_tokens = 200
        
        print("🚀 Initializing BitWhisker 7B...")
        self.model = ModelManager()
        print("🎉 Ready!")
    
    @property
    def current(self) -> ChatThread:
        return self.threads[self.current_index]
    
    def new_thread(self, title: str = "New Chat") -> None:
        self.threads.insert(0, ChatThread(title=title))
        self.current_index = 0
    
    def set_temperature(self, t: float) -> None:
        self.model.temperature = max(0.1, min(float(t), 2.0))
    
    def set_top_p(self, p: float) -> None:
        self.model.top_p = max(0.1, min(float(p), 1.0))
    
    def set_repetition_penalty(self, r: float) -> None:
        self.model.repetition_penalty = max(1.0, min(float(r), 2.0))
    
    def set_max_tokens(self, n: int) -> None:
        self.max_tokens = max(10, min(int(n), 500))
    
    def train_model(self, text: str) -> None:
        self.model.train_on_text(text)


# =====================================================
# UI
# =====================================================

class BitWhiskerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} — Neural LM")
        self.geometry("1100x720")
        self.minsize(920, 600)

        self.theme = tk.StringVar(value="dark")
        self.palette = {
            "dark": {
                "bg": "#0f1115", "bg2": "#151821", "panel": "#10131a",
                "text": "#e8e8e8", "muted": "#9aa2b1", "accent": "#4f8cff",
                "bubble_user": "#1e2533", "bubble_ai": "#151b26", "border": "#1d2230"
            },
            "light": {
                "bg": "#f5f6f8", "bg2": "#ffffff", "panel": "#ffffff",
                "text": "#1f2430", "muted": "#5b6372", "accent": "#2e6bd3",
                "bubble_user": "#e9eef9", "bubble_ai": "#eef1f6", "border": "#dcdfe6"
            }
        }

        self.controller = ChatController()
        self._streaming = False
        self._stop_flag = False
        self._after_token = None
        self._stream_queue: queue.Queue[str | None] = queue.Queue()

        self._build_style()
        self._build_layout()
        self._apply_theme()

    def _build_style(self) -> None:
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure("Sidebar.TFrame", background="#151821")
        self.style.configure("Header.TFrame", background="#0f1115")
        self.style.configure("Footer.TFrame", background="#0f1115")
        self.style.configure("Primary.TButton", padding=8)
        self.style.configure("Flat.TButton", padding=6)

    def _build_layout(self) -> None:
        pal = self._p()

        # Header
        self.header = ttk.Frame(self, style="Header.TFrame")
        self.header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._title_label = tk.Label(
            self.header, text=f"🔥 {APP_NAME}", font=("Arial", 14, "bold"),
            bd=0, padx=12, pady=10
        )
        self._title_label.pack(side=tk.LEFT)
        
        device_label = tk.Label(
            self.header, text=f"[{self.controller.model.device.upper()}]", 
            font=("Arial", 10), bd=0, padx=6, pady=10
        )
        device_label.pack(side=tk.LEFT)

        self._mode_btn = ttk.Button(self.header, text="🌙", 
                                    command=self._toggle_theme, style="Flat.TButton")
        self._mode_btn.pack(side=tk.RIGHT, padx=6, pady=6)

        self._settings_btn = ttk.Button(self.header, text="⚙", 
                                        command=self._open_settings, style="Flat.TButton")
        self._settings_btn.pack(side=tk.RIGHT, padx=6, pady=6)
        
        self._train_btn = ttk.Button(self.header, text="📚 Train", 
                                     command=self._open_training, style="Flat.TButton")
        self._train_btn.pack(side=tk.RIGHT, padx=6, pady=6)

        # Sidebar
        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame")
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=0, minsize=240)

        self._new_btn = ttk.Button(self.sidebar, text="＋ New Chat", 
                                   command=self._new_chat, style="Primary.TButton")
        self._new_btn.pack(fill="x", padx=10, pady=(10, 6))

        self.thread_list = tk.Listbox(self.sidebar, activestyle="dotbox", 
                                      highlightthickness=0, exportselection=False)
        self.thread_list.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.thread_list.bind("<<ListboxSelect>>", self._select_thread)
        self._refresh_thread_list()

        # Main area
        self.main = tk.Frame(self, bd=0, highlightthickness=0)
        self.main.grid(row=1, column=1, sticky="nsew")

        self.canvas = tk.Canvas(self.main, bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.msg_area = tk.Frame(self.canvas)
        self.msg_area.bind("<Configure>", 
                          lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._msg_window = self.canvas.create_window((0, 0), window=self.msg_area, anchor="nw")
        self.canvas.bind("<Configure>", 
                        lambda e: self.canvas.itemconfig(self._msg_window, width=e.width))

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # Footer
        self.footer = ttk.Frame(self, style="Footer.TFrame")
        self.footer.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.grid_rowconfigure(2, weight=0)

        self.input_box = tk.Text(self.footer, height=3, wrap="word", bd=0, padx=12, pady=10)
        self.input_box.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)
        self.input_box.bind("<Return>", self._enter_send)
        self.input_box.bind("<Shift-Return>", lambda e: None)

        self.btn_send = ttk.Button(self.footer, text="Send →", 
                                   command=self._on_send, style="Primary.TButton")
        self.btn_send.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        self.btn_stop = ttk.Button(self.footer, text="Stop", 
                                   command=self._stop_stream, style="Flat.TButton")
        self.btn_stop.grid(row=0, column=2, sticky="nsew", padx=(0, 10), pady=10)

        self.footer.grid_columnconfigure(0, weight=1)
        self.input_box.focus_set()

        params = self.controller.model.model.count_parameters()
        self._add_message("assistant", 
            f"🔥 {APP_NAME} online!\n\n"
            f"I'm a real transformer neural network with {params:,} parameters, "
            f"running on {self.controller.model.device.upper()}. "
            f"I use RoPE embeddings, SwiGLU activations, and nucleus sampling.\n\n"
            f"Ask me anything!")
        self._apply_theme()

    def _p(self) -> dict:
        return self.palette[self.theme.get()]

    def _on_mousewheel(self, event) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _apply_theme(self) -> None:
        pal = self._p()
        self.configure(bg=pal["bg"])
        for fr in (self.main, self.msg_area):
            fr.configure(bg=pal["bg"])
        self.canvas.configure(bg=pal["bg"])
        self._title_label.configure(bg=pal["bg"], fg=pal["text"])
        self.thread_list.configure(
            bg=pal["panel"], fg=pal["text"], selectbackground=pal["accent"],
            selectforeground="#ffffff", bd=0, highlightthickness=0
        )
        self.input_box.configure(
            bg=pal["bg2"], fg=pal["text"], insertbackground=pal["text"], 
            highlightthickness=1, highlightbackground=pal["border"], 
            highlightcolor=pal["accent"]
        )
        for child in self.msg_area.winfo_children():
            self._apply_theme_to_bubble(child, pal)
        self.style.configure("Header.TFrame", background=pal["bg"])
        self.style.configure("Footer.TFrame", background=pal["bg"])
        self.style.configure("Sidebar.TFrame", background=pal["panel"])
        self.update_idletasks()

    def _apply_theme_to_bubble(self, container: tk.Frame, pal: dict) -> None:
        if not hasattr(container, "_bubble_role"):
            return
        role = container._bubble_role
        container.configure(bg=pal["bg"])
        for child in container.winfo_children():
            if isinstance(child, tk.Label):
                try:
                    font = str(child.cget("font"))
                    if "bold" in font:
                        child.configure(bg=pal["bg"], fg=pal["muted"])
                    else:
                        child.configure(fg=pal["text"])
                except:
                    pass
            elif isinstance(child, tk.Frame):
                try:
                    ht = child.cget("highlightthickness")
                    if ht == 1:
                        bubble_bg = pal["bubble_user"] if role == "user" else pal["bubble_ai"]
                        child.configure(bg=bubble_bg, highlightbackground=pal["border"])
                        for lbl in child.winfo_children():
                            if isinstance(lbl, tk.Label):
                                lbl.configure(bg=bubble_bg, fg=pal["text"])
                    else:
                        child.configure(bg=pal["bg"])
                except:
                    child.configure(bg=pal["bg"])

    def _toggle_theme(self) -> None:
        self.theme.set("light" if self.theme.get() == "dark" else "dark")
        self._apply_theme()

    def _new_chat(self) -> None:
        self.controller.new_thread(title="New Chat")
        self._refresh_thread_list()
        self._clear_messages()
        self._add_message("assistant", "Started a new conversation. How can I help?")

    def _refresh_thread_list(self) -> None:
        self.thread_list.delete(0, tk.END)
        for th in self.controller.threads:
            title = th.title if th.title.strip() else "Untitled"
            if len(title) > 32:
                title = title[:29] + "..."
            self.thread_list.insert(tk.END, title)
        self.thread_list.selection_clear(0, tk.END)
        self.thread_list.selection_set(self.controller.current_index)
        self.thread_list.activate(self.controller.current_index)

    def _select_thread(self, event=None) -> None:
        sel = self.thread_list.curselection()
        if not sel:
            return
        self.controller.current_index = sel[0]
        self._render_thread()

    def _render_thread(self) -> None:
        self._clear_messages()
        for msg in self.controller.current.messages:
            self._add_message(msg.role, msg.content, remember=False)
        self._scroll_to_bottom()

    def _clear_messages(self) -> None:
        for child in self.msg_area.winfo_children():
            child.destroy()

    def _add_message(self, role: str, content: str, remember: bool = True) -> None:
        pal = self._p()
        container = tk.Frame(self.msg_area, bg=pal["bg"], padx=12, pady=8)
        container._bubble_role = role
        container.pack(fill="x", anchor="w")

        role_name = "You" if role == "user" else "BitWhisker 7B"
        lbl_role = tk.Label(container, text=role_name, font=("Arial", 9, "bold"),
                            fg=pal["muted"], bg=pal["bg"])
        lbl_role.pack(anchor="w", padx=6)

        bubble_bg = pal["bubble_user"] if role == "user" else pal["bubble_ai"]
        bubble = tk.Frame(container, bg=bubble_bg, bd=0, padx=12, pady=10, 
                         highlightthickness=1)
        bubble.configure(highlightbackground=pal["border"])
        bubble.pack(anchor="w", padx=6, pady=(2, 4), fill="x")

        txt = tk.Label(bubble, text=content, justify="left", anchor="w",
                       font=("Arial", 11), wraplength=750, bg=bubble_bg, fg=pal["text"])
        txt.pack(anchor="w", fill="x")

        tools = tk.Frame(container, bg=pal["bg"])
        tools.pack(anchor="w", padx=6, pady=(0, 6))
        btn_copy = ttk.Button(tools, text="Copy", 
                              command=lambda c=content: self._copy_text(c), 
                              style="Flat.TButton")
        btn_copy.pack(side="left")

        if remember:
            self.controller.current.messages.append(Message(role=role, content=content))

        self.update_idletasks()
        self._scroll_to_bottom()

    def _copy_text(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

    def _scroll_to_bottom(self) -> None:
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _enter_send(self, event) -> str | None:
        if event.state & 0x1:
            return None
        self._on_send()
        return "break"

    def _on_send(self) -> None:
        if self._streaming:
            return
        text = self.input_box.get("1.0", tk.END).strip()
        if not text:
            return
        self.input_box.delete("1.0", tk.END)

        if not self.controller.current.messages:
            self.controller.current.title = text.splitlines()[0][:50]
            self._refresh_thread_list()

        self._add_message("user", text)
        self._stream_assistant(text)

    def _stop_stream(self) -> None:
        self._stop_flag = True

    def _stream_assistant(self, user_text: str) -> None:
        self._stop_flag = False
        self._streaming = True
        pal = self._p()

        container = tk.Frame(self.msg_area, bg=pal["bg"], padx=12, pady=8)
        container._bubble_role = "assistant"
        container.pack(fill="x", anchor="w")

        lbl_role = tk.Label(container, text="BitWhisker 7B", font=("Arial", 9, "bold"),
                            fg=pal["muted"], bg=pal["bg"])
        lbl_role.pack(anchor="w", padx=6)

        bubble_bg = pal["bubble_ai"]
        bubble = tk.Frame(container, bg=bubble_bg, bd=0, padx=12, pady=10, 
                         highlightthickness=1)
        bubble.configure(highlightbackground=pal["border"])
        bubble.pack(anchor="w", padx=6, pady=(2, 4), fill="x")

        txt_var = tk.StringVar(value="")
        txt = tk.Label(bubble, textvariable=txt_var, justify="left", anchor="w",
                       font=("Arial", 11), wraplength=750, bg=bubble_bg, fg=pal["text"])
        txt.pack(anchor="w", fill="x")

        tools = tk.Frame(container, bg=pal["bg"])
        tools.pack(anchor="w", padx=6, pady=(0, 6))
        btn_copy = ttk.Button(tools, text="Copy", 
                              command=lambda: self._copy_text(txt_var.get()), 
                              style="Flat.TButton")
        btn_copy.pack(side="left")

        # Build prompt from conversation
        prompt = ""
        for msg in self.controller.current.messages[-6:]:
            if msg.role == "user":
                prompt += f"User: {msg.content}\n"
            else:
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant:"

        def generate_thread():
            gen = self.controller.model.generate(
                prompt, max_tokens=self.controller.max_tokens,
                stop_event=lambda: self._stop_flag
            )
            for chunk in gen:
                self._stream_queue.put(chunk)
            self._stream_queue.put(None)

        Thread(target=generate_thread, daemon=True).start()

        def poll_queue():
            try:
                while True:
                    chunk = self._stream_queue.get_nowait()
                    if chunk is None:
                        self._finalize_assistant_message(txt_var.get())
                        return
                    txt_var.set(txt_var.get() + chunk)
                    self._scroll_to_bottom()
            except queue.Empty:
                pass
            
            if self._stop_flag:
                self._finalize_assistant_message(txt_var.get())
                return
            
            self._after_token = self.after(20, poll_queue)

        poll_queue()

    def _finalize_assistant_message(self, content: str) -> None:
        self._streaming = False
        self._stop_flag = False
        if self._after_token:
            try:
                self.after_cancel(self._after_token)
            except:
                pass
            self._after_token = None
        while not self._stream_queue.empty():
            try:
                self._stream_queue.get_nowait()
            except queue.Empty:
                break
        content = content.strip()
        if content:
            self.controller.current.messages.append(Message(role="assistant", content=content))

    def _open_settings(self) -> None:
        pal = self._p()
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("450x380")
        win.configure(bg=pal["bg"])

        tk.Label(win, text="⚙ Model Settings", font=("Arial", 12, "bold"), 
                bg=pal["bg"], fg=pal["text"]).pack(pady=(10, 6))

        frm = tk.Frame(win, bg=pal["bg"])
        frm.pack(fill="x", expand=True, padx=16, pady=6)

        # Temperature
        tk.Label(frm, text="Temperature", bg=pal["bg"], 
                fg=pal["text"]).grid(row=0, column=0, sticky="w", pady=4)
        t_var = tk.DoubleVar(value=self.controller.model.temperature)
        tk.Scale(frm, from_=0.1, to=2.0, resolution=0.05, orient="horizontal",
                 variable=t_var, bg=pal["bg"], fg=pal["text"], 
                 highlightthickness=0, troughcolor=pal["bg2"]).grid(row=0, column=1, sticky="ew", padx=8)

        # Top-p
        tk.Label(frm, text="Top-p (nucleus)", bg=pal["bg"], 
                fg=pal["text"]).grid(row=1, column=0, sticky="w", pady=4)
        p_var = tk.DoubleVar(value=self.controller.model.top_p)
        tk.Scale(frm, from_=0.1, to=1.0, resolution=0.02, orient="horizontal",
                 variable=p_var, bg=pal["bg"], fg=pal["text"], 
                 highlightthickness=0, troughcolor=pal["bg2"]).grid(row=1, column=1, sticky="ew", padx=8)

        # Repetition penalty
        tk.Label(frm, text="Repetition penalty", bg=pal["bg"], 
                fg=pal["text"]).grid(row=2, column=0, sticky="w", pady=4)
        r_var = tk.DoubleVar(value=self.controller.model.repetition_penalty)
        tk.Scale(frm, from_=1.0, to=2.0, resolution=0.05, orient="horizontal",
                 variable=r_var, bg=pal["bg"], fg=pal["text"], 
                 highlightthickness=0, troughcolor=pal["bg2"]).grid(row=2, column=1, sticky="ew", padx=8)

        # Max tokens
        tk.Label(frm, text="Max tokens", bg=pal["bg"], 
                fg=pal["text"]).grid(row=3, column=0, sticky="w", pady=4)
        n_var = tk.IntVar(value=self.controller.max_tokens)
        tk.Scale(frm, from_=20, to=500, resolution=10, orient="horizontal",
                 variable=n_var, bg=pal["bg"], fg=pal["text"], 
                 highlightthickness=0, troughcolor=pal["bg2"]).grid(row=3, column=1, sticky="ew", padx=8)

        frm.grid_columnconfigure(1, weight=1)
        
        # Model stats
        params = self.controller.model.model.count_parameters()
        stats_frame = tk.Frame(win, bg=pal["bg"])
        stats_frame.pack(fill="x", padx=16, pady=10)
        tk.Label(stats_frame, 
                text=f"📊 Parameters: {params:,} | Device: {self.controller.model.device.upper()}",
                bg=pal["bg"], fg=pal["muted"], font=("Arial", 9)).pack(anchor="w")
        tk.Label(stats_frame,
                text=f"📐 {self.controller.model.config.n_layers}L / {self.controller.model.config.n_heads}H / {self.controller.model.config.dim}D",
                bg=pal["bg"], fg=pal["muted"], font=("Arial", 9)).pack(anchor="w")

        bar = tk.Frame(win, bg=pal["bg"])
        bar.pack(fill="x", padx=16, pady=10)
        
        def save():
            self.controller.set_temperature(t_var.get())
            self.controller.set_top_p(p_var.get())
            self.controller.set_repetition_penalty(r_var.get())
            self.controller.set_max_tokens(n_var.get())
            win.destroy()
        
        ttk.Button(bar, text="Save", command=save).pack(side="right")
        ttk.Button(bar, text="Save Transcript…", command=self._save_transcript).pack(side="left")

        win.transient(self)
        win.grab_set()

    def _open_training(self) -> None:
        pal = self._p()
        win = tk.Toplevel(self)
        win.title("Fine-tune Model")
        win.geometry("600x450")
        win.configure(bg=pal["bg"])

        tk.Label(win, text="📚 Fine-tune the Neural Network", font=("Arial", 12, "bold"), 
                bg=pal["bg"], fg=pal["text"]).pack(pady=(10, 6))
        tk.Label(win, text="Paste text to train the model on new patterns:", 
                bg=pal["bg"], fg=pal["muted"]).pack(anchor="w", padx=16)

        text_frame = tk.Frame(win, bg=pal["bg"])
        text_frame.pack(fill="both", expand=True, padx=16, pady=10)

        train_text = tk.Text(text_frame, wrap="word", bg=pal["bg2"], fg=pal["text"],
                            insertbackground=pal["text"], highlightthickness=1,
                            highlightbackground=pal["border"])
        train_text.pack(fill="both", expand=True)

        bar = tk.Frame(win, bg=pal["bg"])
        bar.pack(fill="x", padx=16, pady=10)
        
        def do_train():
            text = train_text.get("1.0", tk.END).strip()
            if len(text) < 50:
                messagebox.showwarning("Too short", "Please enter more text (50+ chars).")
                return
            self.controller.train_model(text)
            messagebox.showinfo("Done!", f"Fine-tuned on {len(text)} characters.")
            win.destroy()
        
        def load_file():
            path = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        train_text.delete("1.0", tk.END)
                        train_text.insert("1.0", f.read())
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        ttk.Button(bar, text="Load File…", command=load_file).pack(side="left")
        ttk.Button(bar, text="Train", command=do_train).pack(side="right")
        ttk.Button(bar, text="Cancel", command=win.destroy).pack(side="right", padx=6)

        win.transient(self)
        win.grab_set()

    def _save_transcript(self) -> None:
        th = self.controller.current
        data = th.to_dict()
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile=f"bitwhisker7b_{int(time.time())}.json"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Saved", f"Saved to {path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))


# =====================================================
# Main
# =====================================================

def main():
    app = BitWhiskerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
