from collections import deque, namedtuple
import random
import numpy as np
import torch


replayItem = namedtuple('replayItem', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)


    def append(self, *args) -> None:

        self.states.append(args[0])
        self.actions.append(args[1])
        self.rewards.append(args[2])
        self.next_states.append(args[3])
        self.dones.append(args[4])

    def sample(self, batch_size, device) -> list:
        
        if len(self.states) < batch_size:
            idxs = random.choices(range(len(self.states)), k=batch_size)
        else:
            idxs = random.sample(range(len(self.states)), batch_size)
        
        states = torch.as_tensor(np.array(([self.states[i] for i in idxs]), dtype=np.float32), device=device)
        actions = torch.as_tensor(np.array([self.actions[i] for i in idxs]), device=device)
        rewards = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array(([self.next_states[i] for i in idxs]), dtype=np.float32), device=device)
        dones = torch.tensor([self.dones[i] for i in idxs], dtype=torch.int32, device=device)

        return (states, actions, rewards, next_states, dones)
    
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

class NumpyMemory:
    def __init__(self, capacity, obs_shape, action_shape) -> None:
        self.capacity = capacity
        self.size = 0
        self.idx = 0

        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)

        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.int32)

    def append(self, *args) -> None:
        self.size = min(self.size+1, self.capacity)
        
        self.states[self.idx] = args[0]
        self.actions[self.idx] = args[1]
        self.rewards[self.idx] = args[2]
        self.next_states[self.idx] = args[3]
        self.dones[self.idx] = args[4]
        
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size, device) -> tuple[np.ndarray,...]:
        
        if self.__len__() < batch_size:
            indices = np.random.choice(self.size, batch_size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)

        items = (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices])

        result = tuple(map(lambda x: torch.from_numpy(x).to(device), items))

        return result

    def __len__(self) -> int:
        return self.size
    

class ReplayMemorySlow:
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