"""Convert CRLF to LF for shell scripts (Windows checkout artifact)."""
import sys

for path in sys.argv[1:]:
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data.replace(b"\r\n", b"\n"))
    print("fixed", path)
