"""Keep our files out of the page cache: on the GB10 the kernel does not count GPU memory as memory pressure, so cache
grows until the driver cannot allocate and the node freezes. Every 20 s, POSIX_FADV_DONTNEED every file we touch."""
import glob, os, sys, time
ROOTS = ["/tmp/manors-hf", "/tmp/manors-triton", "/tmp/rm-rank*", "/scratch/scholar/manors/nuance/data",
         "/scratch/scholar/manors/nuance/runs", "/scratch/scholar/manors/nuance/hf"]
while True:
    for r in ROOTS:
        for root in glob.glob(r):
            for d, _, fs in os.walk(root):
                for f in fs:
                    try:
                        fd = os.open(os.path.join(d, f), os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                    except OSError:
                        pass
    if len(sys.argv) > 1:
        break
    time.sleep(20)
