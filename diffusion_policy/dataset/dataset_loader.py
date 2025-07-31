import torch
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.globals import CONFIG


class BatchLoader:
    """
    Responsible for handling requests for batchs of trainable data.
    Uses the torch dataloader to handle data shuffling and querying the underlying sampler.
    """
    def __init__(self):
        self.dataset = <>
        self.dataloader = DataLoader(self.dataset,
            batch_size=dataset_batch_size,
            timeout=self.timeout,
            num_workers=num_workers[idx],
            **cfg.dataloader)
        
        self.device = torch.device(CONFIG.training.device)
        
    def reset(self):
        # reset my count
        self.count = 0

        # forces a reshuffle
        self.iterator = iter(self.dataloader)

    # implicitly called at the start of loops
    def __iter__(self):
        """
        Required for tqdm (progress bar) compatibility
        """

        self.reset()

        return self
    
    def transfer_to_gpu(self, batch):
        batch_gpu = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

        return batch_gpu
    
    def get_batch(self):

        try:
            batch = next(self.iterator) # output: dict
        # except can occur from a timeout or a StopIteration
        # I'm still not sure why the timeout's are occuring, but this will get around it
        except Exception as e:
            print("next(iterator) except: ")
            print(e)
            print("len(dataset): ", len(self.dataset))
            
            # reshuffle this iterator
            self.iterator = iter(self.dataloader)
            
            # get a new batch
            batch = next(self.iterator) # output: dict

        batch_gpu = self.transfer_to_gpu(batch)
        
        # normalizer = dataset.get_normalizer()
        ndata = self.normalize_batch(batch_gpu)
        
        # we're done
        return ndata
    
    # called each for-loop
    def __next__(self):
        
        if self.count < self.num_batches:
            # increment our count
            self.count += 1

            nbatch = self.get_batch()

            return nbatch
        
        else:
            # we've done num_batches
            # print("Iteration done. Count: {}".format(self.count))
            raise StopIteration
