# Motion Detection Demo

This project demonstrates real-time motion detection using OpenCV, implemented
in both single-threaded and multi-process versions.

## Running the Demo

```bash
pipenv run python3 main-single-thread.py
```

```bash
pipenv run python3 main-multi-process.py
```

## Implementations

### Single-threaded Version

The single-threaded implementation (`main-single-thread.py`) processes video
frames synchronously. Despite being single-threaded, it achieves better
performance since OpenCV implements internal concurrency for image processing,
avoiding the overhead of inter-process communication.

Performance: ~200 FPS on the provided sample video (tested on my 2018 MacBook
Pro 15")

### Multi-process Version

The multi-process implementation (`main-multi-process.py`) splits the processing
pipeline into separate processes:

- Video frame streaming
- Motion detection
- Display/UI

While this demonstrates a more complex architecture, it actually runs slower
than the single-threaded version due to the overhead of copying frame data
between processes. A more efficient implementation would require shared memory,
which is outside the scope of this demo.
