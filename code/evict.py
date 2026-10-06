"""Print how much of a file is in the page cache (mincore) and, unless 'report' is given, drop it (posix_fadvise DONTNEED)."""
import ctypes, mmap, os, sys
libc = ctypes.CDLL("libc.so.6", use_errno=True)


def resident(path):
    fd = os.open(path, os.O_RDONLY)
    n = os.fstat(fd).st_size
    mm = mmap.mmap(fd, n, access=mmap.ACCESS_COPY)  # private mapping: pages not written still show the page cache
    os.close(fd)
    first = ctypes.c_char.from_buffer(mm)  # address of the mapping
    page = mmap.PAGESIZE
    vec = (ctypes.c_ubyte * ((n + page - 1) // page))()
    rc = libc.mincore(ctypes.c_void_p(ctypes.addressof(first)), ctypes.c_size_t(n), vec)
    frac = bytes(vec).count(1) / len(vec) if rc == 0 else -1.0  # Linux sets only bit 0
    del first
    mm.close()
    return frac


p = sys.argv[1]
tag = sys.argv[3] if len(sys.argv) > 3 else ""
before = resident(p)
if len(sys.argv) > 2 and sys.argv[2] == "report":
    print(f"PAGECACHE {tag} {100 * before:.1f}%", flush=True)
else:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    print(f"PAGECACHE {tag} {100 * before:.1f}% -> evicted {100 * resident(p):.1f}%", flush=True)
