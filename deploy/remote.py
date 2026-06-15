"""Tiny SSH/SFTP driver (paramiko). Credentials come from env only:
RHOST, RPORT, RUSER, RPASS. Usage:
    python deploy/remote.py run "CMD"
    python deploy/remote.py put  LOCAL_DIR REMOTE_DIR
"""
import os
import sys
import paramiko

HOST = os.environ["RHOST"]
PORT = int(os.environ.get("RPORT", "22"))
USER = os.environ.get("RUSER", "root")
PW = os.environ["RPASS"]
IGNORE = {"__pycache__", ".git", "runs", ".ipynb_checkpoints", ".DS_Store", "deploy"}


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW,
              timeout=40, banner_timeout=40, auth_timeout=40)
    return c


def run(cmd):
    c = client()
    _, stdout, stderr = c.exec_command(cmd, get_pty=False)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    if out:
        print(out, end="")
    if err.strip():
        sys.stderr.write(err)
    c.close()
    sys.exit(rc)


def _mkdirs(sftp, path):
    cur = ""
    for p in path.strip("/").split("/"):
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def put(localdir, remotedir):
    c = client()
    sftp = c.open_sftp()
    n = 0
    for root, dirs, files in os.walk(localdir):
        dirs[:] = [d for d in dirs if d not in IGNORE]
        rel = os.path.relpath(root, localdir)
        rdir = remotedir if rel == "." else os.path.join(remotedir, rel)
        _mkdirs(sftp, rdir)
        for f in files:
            if f.endswith(".pyc") or f == ".DS_Store":
                continue
            sftp.put(os.path.join(root, f), os.path.join(rdir, f))
            n += 1
    print(f"uploaded {n} files to {remotedir}")
    sftp.close()
    c.close()


def get(remotepath, localpath):
    c = client()
    sftp = c.open_sftp()
    os.makedirs(os.path.dirname(localpath) or ".", exist_ok=True)
    sftp.get(remotepath, localpath)
    print(f"downloaded {remotepath} -> {localpath} ({os.path.getsize(localpath)} bytes)")
    sftp.close()
    c.close()


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "put":
        put(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "get":
        get(sys.argv[2], sys.argv[3])
    else:
        sys.exit("unknown subcommand")
