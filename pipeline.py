import multiprocessing

def process_pipeline_step(func, input_queue, output_queue):
    while True:
        data = input_queue.get()
        if data is None:
            break
        for result in func(data):
            output_queue.put(result)

class PipelineStep:
    """A class representing a single step in a processing pipeline.
    
    Each pipeline step runs in its own process and communicates with other steps
    via input and output queues.

    Attributes:
        func: The processing function to run on each input
        input_queue: Queue for receiving input data
        output_queue: Queue for sending processed output
        process: The Process object running this step
    """

    def __init__(self, func, input_queue = None, output_queue = None, cleanup_func = None):
        """Initialize a new pipeline step.

        Args:
            func: Function that processes input data
            input_queue: Queue to receive input from previous step 
            output_queue: Queue to send output to next step
        """
        self.func = func
        self.input_queue = input_queue or multiprocessing.Queue()
        self.output_queue = output_queue or multiprocessing.Queue()
        self.cleanup_func = cleanup_func
        self.process = None

    def start(self):
        """Start the pipeline step process if not already running."""
        if self.process is None:
            self.process = multiprocessing.Process(target=process_pipeline_step, args=(self.func, self.input_queue, self.output_queue))
            self.process.start()

    def stop(self):
        """Stop the pipeline step process and clean up."""
        self.input_queue.put(None)
        self.process.kill()
        self.process = None
        if self.cleanup_func:
            self.cleanup_func()

    def input(self, data):
        """Process a single frame of data.

        Args:
            frame: The data to process
        """
        self.start()
        self.input_queue.put(data)

    def output(self, wait=False):
        try:
            return self.output_queue.get(wait)
        except:
            return None
    
