"""Package a clean Git commit with source metadata; requires only Python stdlib."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="code.tar.gz")
    p.add_argument("--prefix", required=True, choices=["nanochat", "llama2.c"])
    args = p.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
    if subprocess.check_output(["git", "diff", "HEAD", "--", "."], cwd=root):
        raise SystemExit("Commit tracked source changes before creating a reproducible bundle")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    archive = io.BytesIO(subprocess.check_output(["git", "archive", "HEAD"], cwd=root))
    with tarfile.open(args.output, "w:gz") as target, tarfile.open(fileobj=archive) as source:
        for member in source:
            stream = source.extractfile(member) if member.isfile() else None
            member.name = f"{args.prefix}/{member.name}"
            target.addfile(member, stream)
        data = json.dumps(dict(git_commit=commit, dirty=False), indent=2).encode()
        member = tarfile.TarInfo(f"{args.prefix}/gradformer/source.json")
        member.size = len(data)
        target.addfile(member, io.BytesIO(data))
    print(json.dumps(dict(bundle=str(Path(args.output).resolve()), git_commit=commit)))


if __name__ == "__main__":
    main()
