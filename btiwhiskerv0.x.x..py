#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BitWhisker0.910b Pure — Pure Tkinter Edition
A lightweight desktop chat UI with zero pip dependencies.
Features:
  • Dark/Light modes
  • Streaming-style assistant typing
  • New Chat / multi-thread sidebar  
  • Copy message
  • Save transcript
  • Pluggable backend (demo mode included)

Python 3.14+ | Pure tkinter, no external packages.
Run: python bitwhisker_pure.py
"""

import time
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field
from typing import Generator
from threading import Thread
import queue

APP_NAME = "BitWhisker0.910b Pure"


# =====================================================
# Pluggable Backend Interface
# =====================================================

class DemoBackend:
    """
    Demo backend that simulates streaming responses.
    Replace this class with your own API/model integration.
    """
    def __init__(self, temperature: float = 0.9):
        self.temperature = max(0.05, min(float(temperature), 1.5))
        self._responses = [
            "I'm BitWhisker running in pure tkinter mode — no external dependencies! 🐱",
            "This is a demo response. You can plug in any backend: local LLM, API, or custom logic.",
            "The streaming animation you see is built into the UI. Connect your own model to make it real!",
            "Temperature is set to {temp:.2f}. In a real backend, this would affect response randomness.",
            "Try the dark/light theme toggle, create new chats, or save your transcript!",
            "Pure Python 3.14 compatible. Just tkinter. No pip install needed.",
        ]
        self._idx = 0

    def generate(self, messages: list[dict], max_tokens: int = 120, 
                 stop_event: callable = None) -> Generator[str, None, None]:
        """
        Generator yielding text chunks for streaming display.
        Override this method to connect real AI backends.
        """
        # Pick a demo response
        resp = self._responses[self._idx % len(self._responses)]
        resp = resp.format(temp=self.temperature)
        self._idx += 1
        
        # Echo last user message context
        if messages:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
            if last_user:
                resp = f'You said: "{last_user[:50]}{"..." if len(last_user) > 50 else ""}"\n\n{resp}'
        
        # Simulate streaming by yielding chunks
        words = resp.split(" ")
        for i, word in enumerate(words):
            if stop_event and stop_event():
                break
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.03 + (0.02 * (len(word) / 10)))  # variable delay


# =====================================================
# Chat Model/Controller
# =====================================================

@dataclass
class Message:
    role: str  # 'user' | 'assistant' | 'system'
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
        self.temperature = 0.9
        self.max_tokens = 160
        self.model = DemoBackend(temperature=self.temperature)

    @property
    def current(self) -> ChatThread:
        return self.threads[self.current_index]

    def new_thread(self, title: str = "New Chat") -> None:
        self.threads.insert(0, ChatThread(title=title))
        self.current_index = 0

    def set_temperature(self, t: float) -> None:
        self.temperature = max(0.05, min(float(t), 1.5))
        self.model = DemoBackend(temperature=self.temperature)

    def set_max_tokens(self, n: int) -> None:
        self.max_tokens = max(1, min(int(n), 1000))


# =====================================================
# UI
# =====================================================

class BitWhiskerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} — Desktop")
        self.geometry("1100x720")
        self.minsize(920, 600)

        # Theming palettes
        self.theme = tk.StringVar(value="dark")
        self.palette = {
            "dark": {
                "bg": "#0f1115",
                "bg2": "#151821",
                "panel": "#10131a",
                "text": "#e8e8e8",
                "muted": "#9aa2b1",
                "accent": "#4f8cff",
                "bubble_user": "#1e2533",
                "bubble_ai": "#151b26",
                "border": "#1d2230"
            },
            "light": {
                "bg": "#f5f6f8",
                "bg2": "#ffffff",
                "panel": "#ffffff",
                "text": "#1f2430",
                "muted": "#5b6372",
                "accent": "#2e6bd3",
                "bubble_user": "#e9eef9",
                "bubble_ai": "#eef1f6",
                "border": "#dcdfe6"
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

    # --------------------------
    # Style
    # --------------------------
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

    # --------------------------
    # Layout
    # --------------------------
    def _build_layout(self) -> None:
        pal = self._p()

        # Header
        self.header = ttk.Frame(self, style="Header.TFrame")
        self.header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._title_label = tk.Label(
            self.header, text=APP_NAME, font=("Arial", 14, "bold"),
            bd=0, padx=12, pady=10
        )
        self._title_label.pack(side=tk.LEFT)

        self._mode_btn = ttk.Button(self.header, text="Toggle Theme", 
                                    command=self._toggle_theme, style="Flat.TButton")
        self._mode_btn.pack(side=tk.RIGHT, padx=6, pady=6)

        self._settings_btn = ttk.Button(self.header, text="Settings", 
                                        command=self._open_settings, style="Flat.TButton")
        self._settings_btn.pack(side=tk.RIGHT, padx=6, pady=6)

        # Sidebar
        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame")
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=260)

        self._new_btn = ttk.Button(self.sidebar, text="＋  New Chat", 
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

        # Scrollable message area
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

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # Footer input area
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

        # Greeting
        self._add_message("assistant", 
            f"{APP_NAME} is online in demo mode. Pure tkinter — no pip dependencies!")
        self._apply_theme()

    # --------------------------
    # Helpers
    # --------------------------
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

        # Repaint message bubbles
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
                if child.cget("font").endswith("bold"):
                    child.configure(bg=pal["bg"], fg=pal["muted"])
                else:
                    child.configure(fg=pal["text"])
            elif isinstance(child, tk.Frame):
                is_bubble = child.cget("highlightthickness") == 1
                if is_bubble:
                    bubble_bg = pal["bubble_user"] if role == "user" else pal["bubble_ai"]
                    child.configure(bg=bubble_bg, highlightbackground=pal["border"])
                    for lbl in child.winfo_children():
                        if isinstance(lbl, tk.Label):
                            lbl.configure(bg=bubble_bg, fg=pal["text"])
                else:
                    child.configure(bg=pal["bg"])

    def _toggle_theme(self) -> None:
        self.theme.set("light" if self.theme.get() == "dark" else "dark")
        self._apply_theme()

    def _new_chat(self) -> None:
        self.controller.new_thread(title="New Chat")
        self._refresh_thread_list()
        self._clear_messages()
        self._add_message("assistant", f"Started a new conversation in {APP_NAME}.")

    def _refresh_thread_list(self) -> None:
        self.thread_list.delete(0, tk.END)
        for th in self.controller.threads:
            title = th.title if th.title.strip() else "Untitled"
            if len(title) > 36:
                title = title[:33] + "..."
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

        role_name = "You" if role == "user" else "Assistant"
        lbl_role = tk.Label(container, text=role_name, font=("Arial", 9, "bold"),
                            fg=pal["muted"], bg=pal["bg"])
        lbl_role.pack(anchor="w", padx=6)

        bubble_bg = pal["bubble_user"] if role == "user" else pal["bubble_ai"]
        bubble = tk.Frame(container, bg=bubble_bg, bd=0, padx=12, pady=10, 
                         highlightthickness=1)
        bubble.configure(highlightbackground=pal["border"])
        bubble.pack(anchor="w", padx=6, pady=(2, 4), fill="x")

        txt = tk.Label(bubble, text=content, justify="left", anchor="w",
                       font=("Arial", 11), wraplength=820, bg=bubble_bg, fg=pal["text"])
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
        messagebox.showinfo("Copied", "Message copied to clipboard.")

    def _scroll_to_bottom(self) -> None:
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _enter_send(self, event) -> str | None:
        if event.state & 0x1:  # Shift pressed
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
            self.controller.current.title = text.splitlines()[0][:60]
            self._refresh_thread_list()

        self._add_message("user", text)
        self._stream_assistant()

    def _stop_stream(self) -> None:
        self._stop_flag = True

    def _stream_assistant(self) -> None:
        self._stop_flag = False
        self._streaming = True
        pal = self._p()

        container = tk.Frame(self.msg_area, bg=pal["bg"], padx=12, pady=8)
        container._bubble_role = "assistant"
        container.pack(fill="x", anchor="w")

        lbl_role = tk.Label(container, text="Assistant", font=("Arial", 9, "bold"),
                            fg=pal["muted"], bg=pal["bg"])
        lbl_role.pack(anchor="w", padx=6)

        bubble_bg = pal["bubble_ai"]
        bubble = tk.Frame(container, bg=bubble_bg, bd=0, padx=12, pady=10, 
                         highlightthickness=1)
        bubble.configure(highlightbackground=pal["border"])
        bubble.pack(anchor="w", padx=6, pady=(2, 4), fill="x")

        txt_var = tk.StringVar(value="")
        txt = tk.Label(bubble, textvariable=txt_var, justify="left", anchor="w",
                       font=("Arial", 11), wraplength=820, bg=bubble_bg, fg=pal["text"])
        txt.pack(anchor="w", fill="x")

        tools = tk.Frame(container, bg=pal["bg"])
        tools.pack(anchor="w", padx=6, pady=(0, 6))
        btn_copy = ttk.Button(tools, text="Copy", 
                              command=lambda: self._copy_text(txt_var.get()), 
                              style="Flat.TButton")
        btn_copy.pack(side="left")

        messages = [{"role": m.role, "content": m.content} 
                   for m in self.controller.current.messages]

        # Run generation in background thread
        def generate_thread():
            gen = self.controller.model.generate(
                messages, max_tokens=self.controller.max_tokens,
                stop_event=lambda: self._stop_flag
            )
            for chunk in gen:
                self._stream_queue.put(chunk)
            self._stream_queue.put(None)  # Signal done

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
            except Exception:
                pass
            self._after_token = None
        # Drain queue
        while not self._stream_queue.empty():
            try:
                self._stream_queue.get_nowait()
            except queue.Empty:
                break
        self.controller.current.messages.append(Message(role="assistant", content=content))

    # --------------------------
    # Settings / Save
    # --------------------------
    def _open_settings(self) -> None:
        pal = self._p()
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("420x280")
        win.configure(bg=pal["bg"])

        tk.Label(win, text="Settings", font=("Arial", 12, "bold"), 
                bg=pal["bg"], fg=pal["text"]).pack(pady=(10, 6))

        frm = tk.Frame(win, bg=pal["bg"])
        frm.pack(fill="x", expand=True, padx=16, pady=6)

        tk.Label(frm, text="Temperature", bg=pal["bg"], 
                fg=pal["text"]).grid(row=0, column=0, sticky="w", pady=6)
        t_var = tk.DoubleVar(value=self.controller.temperature)
        t_scale = tk.Scale(frm, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                           variable=t_var, bg=pal["bg"], fg=pal["text"], 
                           highlightthickness=0, troughcolor=pal["bg2"])
        t_scale.grid(row=0, column=1, sticky="ew", padx=8)

        tk.Label(frm, text="Max tokens", bg=pal["bg"], 
                fg=pal["text"]).grid(row=1, column=0, sticky="w", pady=6)
        n_var = tk.IntVar(value=self.controller.max_tokens)
        n_scale = tk.Scale(frm, from_=16, to=1000, resolution=1, orient="horizontal",
                           variable=n_var, bg=pal["bg"], fg=pal["text"], 
                           highlightthickness=0, troughcolor=pal["bg2"])
        n_scale.grid(row=1, column=1, sticky="ew", padx=8)

        frm.grid_columnconfigure(1, weight=1)

        bar = tk.Frame(win, bg=pal["bg"])
        bar.pack(fill="x", padx=16, pady=10)
        ttk.Button(bar, text="Save", 
                  command=lambda: self._apply_settings(t_var.get(), n_var.get(), win)
                  ).pack(side="right")
        ttk.Button(bar, text="Save Transcript…", 
                  command=self._save_transcript).pack(side="left")

        win.transient(self)
        win.grab_set()
        win.focus_set()

    def _apply_settings(self, temp: float, max_tokens: int, win: tk.Toplevel) -> None:
        self.controller.set_temperature(temp)
        self.controller.set_max_tokens(max_tokens)
        win.destroy()
        messagebox.showinfo("Settings", "Updated generation settings.")

    def _save_transcript(self) -> None:
        th = self.controller.current
        data = th.to_dict()
        default_name = f"{APP_NAME.replace(' ', '_').lower()}_{int(time.time())}.json"
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_name
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Transcript saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{e}")


# =====================================================
# Main
# =====================================================

def main():
    app = BitWhiskerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
