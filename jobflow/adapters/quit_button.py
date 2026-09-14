"""An always-on-top Quit button for a running session.

A browser-driving run takes over the screen, so the terminal that started it
is often buried. This puts a small window on top of everything with one
button, and sets an Event when it is pressed.

The Event is polled by the runner between postings, never mid-application:
stopping halfway through a submission would leave a half-filled form in an
unknown state. Quitting is therefore always "finish this one, then stop".

Tkinter must own the thread it was created on, so the window runs on its own
thread with its own event loop and communicates only through the Event.

Author: Jashan Sadioura
"""

from __future__ import annotations

import atexit
import logging
import threading

log = logging.getLogger(__name__)


class QuitButton:
    """A floating Quit button backed by a threading.Event.

    Degrades to a no-op when no display is available (a headless machine, or
    a Tk that fails to start), because a missing button must never stop a
    run from working.
    """

    def __init__(self, title: str = "JobFlow") -> None:
        self._event = threading.Event()
        self._thread: threading.Thread | None = None
        self._root = None
        self._title = title
        self._closing = threading.Event()

    # -- public surface --------------------------------------------------

    @property
    def quit_requested(self) -> bool:
        return self._event.is_set()

    def request_quit(self) -> None:
        """Signal a quit. Exposed so tests and callers can trigger it."""
        self._event.set()

    def start(self) -> "QuitButton":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="jobflow-quit-button")
        self._thread.start()
        # A daemon thread still inside mainloop when the interpreter exits
        # leaves a live Tk for the main thread to finalize, which aborts with
        # "Tcl_AsyncDelete: async handler deleted by the wrong thread" and a
        # non-zero exit code -- turning a clean run into an apparent failure.
        # Tearing down here guarantees the window is gone first, even when a
        # caller forgets to stop() or an exception skips it.
        atexit.register(self.stop)
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Close the window. Safe to call when it never opened.

        Destruction is left to the window's own thread via the `_closing`
        flag its poll loop watches. Calling into Tk from this thread --
        even `after` -- trips "Tcl_AsyncDelete: async handler deleted by
        the wrong thread", because Tk objects must be torn down on the
        thread that created them.
        """
        self._closing.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # -- window ----------------------------------------------------------

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as e:
            log.debug("Quit button unavailable (no tkinter): %s", e)
            return

        try:
            root = tk.Tk()
        except Exception as e:
            # No display, or Tk refused to start. The run continues without
            # a button; Ctrl+C in the terminal still works.
            log.debug("Quit button unavailable (no display): %s", e)
            return

        self._root = root
        root.title(self._title)
        root.attributes("-topmost", True)
        root.resizable(False, False)
        root.geometry("240x110+40+40")
        # The window manager's X does the same thing as the button: a person
        # closing it means "stop", not "keep running invisibly".
        root.protocol("WM_DELETE_WINDOW", self._on_click)

        tk.Label(root, text="JobFlow is running",
                 font=("Segoe UI", 10, "bold")).pack(pady=(14, 2))
        status = tk.Label(root, text="Click Quit to stop after\nthe current application.",
                          font=("Segoe UI", 8), justify="center")
        status.pack()
        tk.Button(root, text="Quit", width=12, command=self._on_click).pack(pady=8)

        def poll() -> None:
            # Lets stop() close the window even while mainloop owns the thread.
            if self._closing.is_set():
                try:
                    # quit() first: it ends mainloop and releases Tk's async
                    # handler on this thread. Without it the handler survives
                    # to be collected by the main thread at interpreter exit,
                    # which prints "Tcl_AsyncDelete: async handler deleted by
                    # the wrong thread".
                    root.quit()
                    root.destroy()
                except Exception:
                    pass
                return
            if self._event.is_set():
                status.config(text="Stopping after the\ncurrent application...")
            root.after(200, poll)

        root.after(200, poll)
        try:
            root.mainloop()
        except Exception:
            pass
        finally:
            self._root = None

    def _on_click(self) -> None:
        self._event.set()
        log.info("")
        log.info("Quit requested -- stopping after the current application.")


def show_summary(lines: list[str], title: str = "JobFlow - run summary") -> bool:
    """Show the closing summary in a window. True if it was displayed.

    Returns False when there is no display, so the caller can fall back to
    printing: a summary that cannot be shown must never be lost.

    This blocks until the window is closed, and is therefore called after
    the run has finished rather than during it.
    """
    try:
        import tkinter as tk
    except Exception as e:
        log.debug("Summary window unavailable (no tkinter): %s", e)
        return False

    try:
        root = tk.Tk()
    except Exception as e:
        log.debug("Summary window unavailable (no display): %s", e)
        return False

    try:
        root.title(title)
        root.attributes("-topmost", True)
        text = "\n".join(lines).strip("\n")

        frame = tk.Frame(root, padx=16, pady=12)
        frame.pack(fill="both", expand=True)

        # Monospaced: the summary is column-aligned, and a proportional
        # font would throw every number out of line.
        widget = tk.Text(frame, width=74, height=min(30, len(lines) + 2),
                         font=("Consolas", 9), wrap="none",
                         borderwidth=0, background=root.cget("background"))
        widget.insert("1.0", text)
        widget.config(state="disabled")
        widget.pack(fill="both", expand=True)

        tk.Button(frame, text="Close", width=14,
                  command=root.destroy).pack(pady=(10, 0))

        root.update_idletasks()
        root.mainloop()
        return True
    except Exception as e:
        log.debug("Could not render the summary window: %s", e)
        try:
            root.destroy()
        except Exception:
            pass
        return False


class NullQuitButton:
    """Stand-in for runs with no GUI: never requests a quit."""

    quit_requested = False

    def request_quit(self) -> None:  # pragma: no cover - trivial
        pass

    def start(self) -> "NullQuitButton":
        return self

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
#
# Thanks for using JobFlow.
#
# Built because an application carries your name, so the tool should stop and
# ask rather than guess. If it helped your search, pass it on to someone else
# who is looking.
#
# Jashan Sadioura  ·  https://www.linkedin.com/in/sadioura-jashan/
#
# ---------------------------------------------------------------------------
