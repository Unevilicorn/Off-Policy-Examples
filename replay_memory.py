from collections import deque, namedtuple
import random
import numpy as np
import torch


replayItem = namedtuple('replayItem', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.memory = deque(maxlen=capacity)

    def append(self, *args) -> None:
        self.memory.append(replayItem(*args))

    def sample(self, batch_size, device) -> list:
        if len(self.memory) < batch_size:
            samples = [random.choice(self.memory) for _ in range(batch_size)]
        else:
            samples = random.sample(self.memory, batch_size)

        reshaped_items = tuple(map(np.stack, zip(*samples)))
        reshaped_items = tuple([
            x.astype(np.float32) if x.dtype == np.float64 else x for x in reshaped_items
        ])
        result = tuple(map(lambda x: torch.from_numpy(x).to(device), reshaped_items))

        return result
    
    def __len__(self) -> int:
        return len(self.memory)

class ReverbMemory:
    def __init__(self, capacity) -> None:
        import reverb
        self.table_name = 'priority_table'
        self.server = reverb.Server(tables=[
            reverb.Table(
                name=self.table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=capacity,
                rate_limiter=reverb.rate_limiters.MinSize(1),
            )
        ])
        self.writer_client = reverb.client.Client(f'localhost:{self.server.port}')
        self.reader_client = reverb.client.Client(f'localhost:{self.server.port}')

    def append(self, *args) -> None:
        self.writer_client.insert(replayItem(*args), priorities={self.table_name: 1.0})

    def sample(self, batch_size, device) -> tuple[np.ndarray,...]:
        samples = self.reader_client.sample(self.table_name, num_samples=batch_size)
        result = [sample[0].data for sample in samples]

        reshaped_items = tuple(map(np.stack, zip(*result)))
        result = tuple(map(lambda x: torch.from_numpy(x).to(device), reshaped_items))

        return result

    def __len__(self) -> int:
        return self.writer_client.server_info()['priority_table'].current_size
