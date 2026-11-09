"""TEST ONLY: a minimal job-control shell on a pseudo-terminal (tests/test_job_control.py).

    python pty_shell.py CTL_FD EVT_FD {fg,bg,zsh} [PASS_FD ...] -- CHILD ARGV...

Run with the pty slave on fds 0-2 and ``start_new_session=True``. Like an interactive shell it
makes the slave its controlling terminal, ignores the job-control signals itself, and starts the
child in a process group of its own: ``fg`` hands that group the terminal's foreground
(``tcsetpgrp``), ``bg`` leaves it in the background (``cmd &``), ``zsh`` is ``fg`` that also takes
the foreground back as soon as its child stops, as zsh / bash do ("suspended"). The child's group
is NOT orphaned (its parent, this shell, is in another group of the same session), so a Ctrl-Z
typed on the pty really stops it unless it handles SIGTSTP: the negative control proves that.

Commands on CTL_FD, one per line: ``TAKE`` (the shell takes the foreground back, as after a job
stopped or a ``bg``), ``GIVE`` (the child gets it again), ``CONT`` (SIGCONT to the child's group).
Events on EVT_FD: ``STARTED <pid>``, ``STOPPED <signal>``, ``EXIT <code>``. EOF on CTL_FD (the
test ended) kills the child's group; the shell exits when the child does.
"""

from __future__ import annotations

import fcntl
import os
import select
import signal
import subprocess
import sys
import termios


def main() -> int:
    sep = sys.argv.index("--")
    ctl, evt, mode, *keep = sys.argv[1:sep]
    argv = sys.argv[sep + 1 :]
    ctl_fd, evt_fd = int(ctl), int(evt)
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    # the child first: an ignored signal stays ignored across exec, and the child must get the
    # default dispositions, as from a real shell (the negative control checks it)
    child = subprocess.Popen(argv, process_group=0, pass_fds=[int(k) for k in keep])
    pid = child.pid
    for s in (signal.SIGTSTP, signal.SIGTTIN, signal.SIGTTOU, signal.SIGINT, signal.SIGQUIT):
        signal.signal(s, signal.SIG_IGN)
    if mode in ("fg", "zsh"):
        os.tcsetpgrp(0, pid)

    def event(line: str) -> None:
        os.write(evt_fd, (line + "\n").encode())

    event(f"STARTED {pid}")
    buf = b""
    while True:
        ready = select.select([ctl_fd], [], [], 0.02)[0]
        if ready:
            chunk = os.read(ctl_fd, 1024)
            if not chunk:  # the test is gone: never leave a child behind
                os.killpg(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return 0
            buf += chunk
            while b"\n" in buf:
                cmd, buf = buf.split(b"\n", 1)
                if cmd == b"TAKE":
                    os.tcsetpgrp(0, os.getpgrp())
                elif cmd == b"GIVE":
                    os.tcsetpgrp(0, pid)
                elif cmd == b"CONT":
                    os.killpg(pid, signal.SIGCONT)
        got, status = os.waitpid(pid, os.WUNTRACED | os.WNOHANG)
        if got == 0:
            continue
        if os.WIFSTOPPED(status):
            if mode == "zsh":
                os.tcsetpgrp(0, os.getpgrp())
            event(f"STOPPED {signal.Signals(os.WSTOPSIG(status)).name}")
            continue
        event(f"EXIT {os.waitstatus_to_exitcode(status)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
