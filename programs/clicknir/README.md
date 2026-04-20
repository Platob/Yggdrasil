# Clicknir

A small, cross-platform auto mouse clicker with a visual control panel.
Written in Rust with [`eframe`/`egui`](https://github.com/emilk/egui) for the UI
and [`enigo`](https://github.com/enigo-rs/enigo) for synthetic input.

This crate is completely isolated from the rest of the repository — it has its
own `Cargo.toml` and does not participate in the workspace.

## Features

- Clicks-per-second slider (0.5 – 100 cps, logarithmic)
- Left / middle / right button selection
- Single or double click
- Fixed click count or run until stopped
- Configurable start delay so you can position the cursor
- Start / Stop via the UI button or the **F6** hotkey (while the window is focused)
- Live click counter
- Remembers your last configuration between launches

## Build & run

Clicknir targets Rust 1.74+.

```bash
cd programs/clicknir
cargo run --release
```

### Linux prerequisites

`eframe` needs a working display server and a few native libraries. On
Debian/Ubuntu:

```bash
sudo apt install \
    libxcb1-dev libxrandr-dev libxi-dev libxkbcommon-dev \
    libwayland-dev libgl1-mesa-dev libxdo-dev
```

`libxdo-dev` is required by `enigo` for synthesizing mouse input on X11.

`enigo` uses X11 by default; if you are on Wayland, run under XWayland or set
`WINIT_UNIX_BACKEND=x11`.

### Windows

No extra setup — `cargo run --release` is enough. The release binary is a
windowed app (no console window).

## Usage

1. Launch the app.
2. Set the click rate, button, action, and (optional) click count.
3. Set a start delay (default 3 s) so you have time to move the cursor.
4. Press **Start** (or F6). Clicking begins at the current cursor location
   after the delay.
5. Press **Stop** (or F6) to halt.

## Safety

Auto-clickers can be flagged by anti-cheat software or violate site /
application terms of service. Use on software you own or are explicitly
permitted to automate.
